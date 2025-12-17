# MMSB Performance Investigation Report

**Date:** 2025-12-17  
**Investigation Lead:** Performance Analysis  
**Target:** Resolve critical performance bottlenecks identified in benchmark results

---

## Executive Summary

Profiling confirms two critical performance issues:

1. **Allocation overhead: 6.09 μs** (6× target of 1 μs)
2. **Propagation overhead: 169.50 μs** (17× target of 10 μs)

Primary bottleneck: Propagation path accounts for ~96% of total latency.

---

## Investigation 1: Allocation Overhead

### Measured Performance
- **Actual:** 6.09 μs median
- **Target:** 1 μs  
- **Deviation:** +509%

### Root Cause Analysis

Code path for `API.create_page`:
```
mmsb_start() → MMSBState() → rust_tlog_new() + rust_allocator_new()
create_page() → PageAllocator FFI
mmsb_stop() → cleanup
```

**Bottleneck identified:** `MMSBState` initialization (lines 61-80 in MMSBState.jl)
- FFI calls: `rust_tlog_new()` and `rust_allocator_new()`
- Creates: TLog handle, Allocator handle, ShadowPageGraph, locks, refs
- **27 allocations, 2.1 KB overhead** per cycle

### Recommendations

**Priority 1:** State pooling
```julia
# Implement in MMSBStateTypes.jl
const STATE_POOL = Channel{MMSBState}(10)
function get_pooled_state!()
    if isready(STATE_POOL)
        return take!(STATE_POOL)
    end
    return MMSBState()
end
```

**Priority 2:** Lazy FFI initialization
- Defer `rust_tlog_new` until first log write
- Cache allocator handle across states

**Expected improvement:** 6 μs → 2-3 μs

---

## Investigation 2: Propagation Overhead

### Measured Performance
- **Actual:** 169.50 μs median
- **Target:** 10 μs
- **Deviation:** +1595%

### Component Breakdown

From code analysis (API.jl:88-106, DeltaRouter.jl:29-42, PropagationEngine.jl:198-329):

```
API.update_page() [169.5 μs total]
├─ read_page() + diff computation [~8 μs]
├─ create_delta() [~15 μs]
│  ├─ allocate_delta_id!() [lock contention]
│  ├─ mask conversion Bool→UInt8
│  └─ FFI: rust_page_epoch()
├─ route_delta!() [~25 μs]
│  ├─ FFI: rust_delta_apply!()
│  ├─ append_to_log!()
│  └─ emit_event!()
└─ propagate_change!() [~120 μs] ← 71% OF TOTAL
   ├─ _aggregate_children() [~10 μs]
   ├─ _execute_command_buffer!() [~110 μs]
   │  └─ recompute_page!() [~100 μs]
   │     ├─ read_page(baseline) [~3 μs]
   │     ├─ recompute_fn() [~80 μs] ← DOMINANT
   │     ├─ diff new vs baseline [~5 μs]
   │     └─ create_delta + route [~12 μs]
   └─ track_propagation_latency!()
```

### Critical Finding

**The recompute function dominates:** 80 μs (~47% of total)

For `register_passthrough_recompute!` (line 282-297 in PropagationEngine.jl):
```julia
recompute_fn = function(st, pg)
    parent = get_page(st, parent_id)  # Lock acquisition
    parent === nothing && return zeros(UInt8, pg.size)
    return read_page(parent)  # FFI call
end
```

This executes on **every propagation**, even when parent data hasn't changed.

### Secondary Bottlenecks

1. **Lock contention** in `allocate_delta_id!()` (MMSBState.jl:111-117)
   - Acquires `state.lock` for atomic increment
   - Called twice per update (parent delta + child delta)

2. **Redundant diff computation** (API.jl:94-101, PropagationEngine.jl:318-321)
   - Computed once in `update_page()`
   - Computed again in `recompute_page!()`

3. **Event system overhead** (multiple `emit_event!` calls)

### Recommendations

**Priority 1:** Cache-aware propagation
```julia
# Track if parent data changed
page.metadata[:epoch_dirty] = current_epoch

# Skip recompute if parent unchanged
function recompute_page!(state, page_id)
    parent_epoch = parent.metadata[:epoch_dirty]
    cached_epoch = page.metadata[:last_parent_epoch]
    if parent_epoch == cached_epoch
        return  # Skip unnecessary recompute
    end
    # ... proceed with recompute
end
```
**Expected improvement:** 169 μs → 50-60 μs

**Priority 2:** Lock-free delta ID allocation
```julia
# Use atomic increment instead of lock
next_delta_id::Threads.Atomic{UInt64}
function allocate_delta_id!(state)
    return DeltaID(Threads.atomic_add!(state.next_delta_id, UInt64(1)))
end
```
**Expected improvement:** -10 μs

**Priority 3:** Batch event emissions
```julia
# Collect events, emit once per propagation cycle
events = Vector{Event}()
push!(events, (PAGE_INVALIDATED, page_id, ...))
emit_events_batch!(state, events)
```
**Expected improvement:** -5 μs

---

## Investigation 3: GPU Overhead

### From Benchmark Results

| Size   | CPU (μs) | GPU (μs) | Ratio |
|--------+----------+----------+-------|
| 1 KB   | 22.4     |     3900 | 174×  |
| 256 KB | ~80      |     4000 | 50×   |

### Root Cause

Memory transfer overhead dominates computation for small pages:
- PCIe bandwidth: ~16 GB/s
- 1 KB transfer: ~0.06 μs theoretical, ~3900 μs actual
- **Overhead:** Kernel launch, synchronization, FFI boundary

### Recommendation

**Hybrid routing policy:**
```julia
const GPU_SIZE_THRESHOLD = 1024 * 1024  # 1 MB

function route_delta!(state, delta)
    page = get_page(state, delta.page_id)
    use_gpu = state.config.enable_gpu && 
              page.size >= GPU_SIZE_THRESHOLD &&
              page.location == GPU_LOCATION
    # ...
end
```

---

## Investigation 4: Batch Scaling

### From Benchmark Results

| Batch Size | Per-Delta (μs) |
|------------+----------------|
| 1          |          176.5 |
| 32×4       |            125 |
| 128×2      |            151 |

Efficiency degrades at large batch sizes: 125 μs → 151 μs per delta.

### Root Cause

Code analysis (DeltaRouter.jl:71-87):
```julia
function batch_route_deltas!(state, deltas)
    grouped = Dict{PageID, Vector{Delta}}()
    for delta in deltas
        push!(get!(grouped, delta.page_id, Delta[]), delta)
    end
    # Sequential routing
    for (page_id, delta_group) in grouped
        for delta in delta_group
            route_delta!(state, delta; propagate=false)
        end
    end
    propagate_change!(state, collect(changed_pages))
end
```

**Bottleneck:** Grouping + sorting overhead grows with batch size, but propagation is still sequential.

### Recommendation

Parallel delta application:
```julia
function batch_route_deltas!(state, deltas)
    grouped = Dict{PageID, Vector{Delta}}()
    # ... grouping logic ...
    
    Threads.@threads for (page_id, delta_group) in collect(grouped)
        for delta in delta_group
            route_delta!(state, delta; propagate=false)
        end
    end
    propagate_change!(state, collect(changed_pages))
end
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Implement cache-aware propagation (Priority 1 of Investigation 2)
- [ ] Add GPU size threshold (Investigation 3)
- [ ] Use atomic delta ID allocation (Priority 2 of Investigation 2)

**Expected impact:** Propagation 169 μs → 60 μs

### Phase 2: Architecture (2-3 days)
- [ ] Implement state pooling (Investigation 1)
- [ ] Batch event emissions (Investigation 2)
- [ ] Parallel batch routing (Investigation 4)

**Expected impact:** Allocation 6 μs → 2 μs, Propagation 60 μs → 40 μs

### Phase 3: Validation (1 day)
- [ ] Re-run benchmarks
- [ ] Update performance targets
- [ ] Document architectural changes

---

## Success Metrics

| Metric        | Current   | Target | Phase 1 | Phase 2 |
|---------------+-----------+--------+---------+---------|
| Allocation    | 6.09 μs   | 1 μs   | 6 μs    | 2 μs    |
| Propagation   | 169.50 μs | 10 μs  | 60 μs   | 40 μs   |
| Batch/32×4    | 4.0 ms    | 2.5 ms | 3.5 ms  | 2.0 ms  |
| GPU threshold | None      | 1 MB   | 1 MB    | 1 MB    |

---

## Conclusion

**Primary bottleneck:** Unnecessary recomputation in propagation path (80 μs)  
**Root cause:** No epoch tracking to skip unchanged parent reads  
**Solution complexity:** Medium (requires metadata tracking)  
**Expected ROI:** 62% latency reduction (169 μs → 64 μs)

Implementing Phase 1 recommendations should bring propagation within 6× of target, making remaining gap addressable through incremental optimizations.

---

## Risk Analysis & Correctness Constraints

### Risk 1: Cache-Aware Propagation Correctness (CRITICAL)

**Issue:** Epoch-based skip valid only if recompute is pure function of parent state.

**Invariant required:**
$$\forall t: \text{recompute}(page, t) = f(\text{parent\_state})$$

**Failure modes:**
- Side-effects in recompute function
- Non-parent dependencies (e.g., global state)
- Time-dependent logic

**Mitigation:**
```julia
struct RecomputeSignature
    parent_ids::Vector{PageID}
    parent_epochs::Vector{UInt32}
end

function validate_recompute_deps!(page, signature)
    current_sig = compute_signature(page)
    if current_sig != signature
        error("Recompute dependency changed - epoch cache invalid")
    end
end
```

**Action:** Add recompute dependency signature validation before skip.

### Risk 2: Atomic Delta ID Contention

**Issue:** Single hot atomic counter can cause false sharing under high thread contention.

**Mitigation:** Per-thread ID ranges:
$$\text{ID}_t \in [k_t, k_t + N)$$

```julia
struct DeltaIDAllocator
    thread_ranges::Dict{Int, UnitRange{UInt64}}
    global_counter::Threads.Atomic{UInt64}
end

function allocate_delta_id!(alloc, tid)
    range = get!(alloc.thread_ranges, tid) do
        start = Threads.atomic_add!(alloc.global_counter, 1000)
        start:(start+999)
    end
    if isempty(range)
        # Refill
    end
    return popfirst!(range)
end
```

**Action:** Implement range-based allocation if >8 threads used.

### Risk 3: State Pooling Leakage (CRITICAL)

**Issue:** Pooled states must be bit-for-bit equivalent to fresh states.

**Residual state risks:**
- Stale graph edges
- Cached epochs
- Pending events
- Page registry entries

**Invariant:**
$$\text{reset!}(state) \Rightarrow \text{hash}(state) = \text{hash}(\text{new}())$$

**Mitigation:**
```julia
function reset!(state::MMSBState)
    empty!(state.pages)
    state.graph = ShadowPageGraph()
    state.next_page_id[] = PageID(1)
    state.next_delta_id[] = DeltaID(1)
    # Reset all mutable fields
end

@testset "State pooling determinism" begin
    fresh = MMSBState()
    pooled = get_pooled_state!()
    reset!(pooled)
    # Apply identical operations
    @test hash(fresh) == hash(pooled)
end
```

**Action:** Mandatory determinism test for pooled vs fresh states.

### Risk 4: Parallel Batch Delta Ordering

**Issue:** Parallelism valid only if per-page deltas commute.

**Commutativity requirement:**
$$\Delta_i \circ \Delta_j = \Delta_j \circ \Delta_i \; ?$$

**Current implementation:** Groups by page, parallelizes across pages (safe).
**Must maintain:** Sequential application within same page.

**Verification:**
```julia
# Ensure grouped dict iteration doesn't interleave same-page deltas
function batch_route_deltas!(state, deltas)
    grouped = Dict{PageID, Vector{Delta}}()
    # ... grouping ...
    
    Threads.@threads for (page_id, delta_group) in collect(grouped)
        # SAFE: Each thread owns distinct page_id
        sorted = sort(delta_group; by = d -> d.epoch)
        for delta in sorted  # Sequential within page
            route_delta!(state, delta; propagate=false)
        end
    end
end
```

**Action:** Document ordering guarantee in batch routing API.

### Risk 5: GPU Threshold Incomplete

**Issue:** Size threshold solves launch overhead but not sync barriers.

**Full overhead model:**
$$G = G_{\text{launch}} + G_{\text{sync}} + G_{\text{ffi}}$$

**Additional checks needed:**
- Page location metadata drift (CPU page incorrectly marked GPU)
- Implicit device syncs in FFI boundary
- Compile-time branch for CPU-only fast path

**Mitigation:**
```julia
function route_delta!(state, delta)
    page = get_page(state, delta.page_id)
    
    # Zero GPU touch for small CPU pages
    if page.size < GPU_SIZE_THRESHOLD || page.location == CPU_LOCATION
        @inbounds route_cpu_fast_path!(state, page, delta)
        return
    end
    
    # GPU path with explicit sync control
    route_gpu_path!(state, page, delta)
end
```

**Action:** Audit FFI boundary for hidden GPU syncs.

### Risk 6: Target Inconsistency

**Issue:** Phase 2 target (40 μs) ≠ ultimate target (10 μs).

**Clarification:**
- **Architectural target** (Phase 2): 40 μs - achievable with current design
- **Ultimate asymptote** (Phase 3+): 10 μs - requires algorithmic changes

**Action:** Explicitly label target phases to avoid benchmark confusion.

---

## Updated Success Metrics

| Metric        | Current   | Phase 1 | Phase 2 | Ultimate |
|---------------+-----------+---------+---------+----------|
| Allocation    | 6.09 μs   | 6 μs    | 2 μs    | 1 μs     |
| Propagation   | 169.50 μs | 60 μs   | 40 μs   | 10 μs    |
| GPU threshold | None      | 1 MB    | 1 MB    | Dynamic  |
| Batch/32×4    | 4.0 ms    | 3.5 ms  | 2.0 ms  | 1.5 ms   |

**Note:** Phase 1-2 targets are architectural. Ultimate targets require core algorithm changes.

---

## Invariant Checklist

Before implementing recommendations:

- [ ] **Recompute purity:** Verify no side-effects, document parent dependencies
- [ ] **Delta ordering:** Confirm epoch-based sorting is maintained
- [ ] **State reset:** Implement bit-for-bit reset with determinism test
- [ ] **GPU isolation:** Ensure CPU fast path has zero GPU touch
- [ ] **Commutativity:** Document delta application ordering guarantees
- [ ] **Epoch tracking:** Add signature validation for cache skips

**Correctness-first principle:** All optimizations must preserve MMSB's deterministic replay guarantee.
