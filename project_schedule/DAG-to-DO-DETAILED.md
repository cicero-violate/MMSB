## MMSB Performance Remediation DAG (DETAILED)

### Level 0 — Invariants & Ground Truth (FOUNDATION)

**T0.1 — Define Epoch Semantics**

* Specify exact meaning of `epoch` (per-page content epoch vs global tick)
* Decide scope: parent content change indicator

**Solution:**
```julia
# Epoch is per-page content version
struct Page
    epoch::UInt32  # Increments on every delta application
end
# Track when content last changed
page.metadata[:epoch_dirty] = current_epoch
```
**Status:** ⚠️ Needs formal specification

**T0.2 — Define Recompute Dependency Contract**

* Explicitly declare recompute dependencies (parent IDs only)
* Prohibit side-effects and global state usage

**Solution:**
```julia
function register_passthrough_recompute!(state, child_id, parent_id)
    page = get_page(state, child_id)
    page.metadata[:recompute_fn] = function(st, pg)
        parent = get_page(st, parent_id)
        parent === nothing && return zeros(UInt8, pg.size)
        return read_page(parent)  # Pure function
    end
    page.metadata[:recompute_deps] = [parent_id]
end
```
**Status:** ✅ Current implementation is pure

**T0.3 — Define ID Scope**

* Decide delta/page ID uniqueness domain

**Solution:** Per-state scope. IDs unique within MMSBState lifetime, reset on state creation.
**Invariant:** Replay from checkpoint must generate identical IDs
**Status:** ✅ Documented

---

### Level 1 — Correctness Instrumentation (BLOCKING)

**T1.1 — Recompute Dependency Signature System**

**Solution:**
```julia
struct RecomputeSignature
    parent_ids::Vector{PageID}
    parent_epochs::Vector{UInt32}
end

function compute_signature(state, page)
    deps = get(page.metadata, :recompute_deps, PageID[])
    epochs = map(deps) do parent_id
        p = get_page(state, parent_id)
        p === nothing ? UInt32(0) : p.metadata[:epoch_dirty]
    end
    return RecomputeSignature(deps, epochs)
end
```
**File:** `src/04_propagation/PropagationEngine.jl`
**Status:** 🔨 To implement

**T1.2 — Recompute Purity Validation**

**Solution:**
```julia
function recompute_page!(state, page_id)
    page = get_page(state, page_id)
    page === nothing && return
    
    current_sig = compute_signature(state, page)
    cached_sig = get(page.metadata, :last_signature, nothing)
    
    if cached_sig !== nothing
        if current_sig.parent_ids != cached_sig.parent_ids
            error("Dependency changed - cache invalid for page $(page_id)")
        end
        if current_sig.parent_epochs == cached_sig.parent_epochs
            return  # Skip recompute
        end
    end
    
    # Perform recompute
    recompute_fn = get(page.metadata, :recompute_fn, nothing)
    recompute_fn === nothing && return
    baseline = read_page(page)
    new_data = recompute_fn(state, page)
    
    mask = Vector{Bool}(undef, page.size)
    @inbounds for i in eachindex(baseline)
        mask[i] = baseline[i] != new_data[i]
    end
    any(mask) || return
    
    delta = create_delta(state, page_id, mask, new_data; source=:propagation)
    route_delta!(state, delta)
    page.metadata[:last_signature] = current_sig
end
```
**File:** `src/04_propagation/PropagationEngine.jl:310`
**Expected improvement:** 169 μs → 90 μs (skip 80 μs recompute)
**Status:** 🔨 To implement

**T1.3 — Deterministic Replay Oracle**

**Solution:**
```julia
function canonical_snapshot(state)
    return Dict(
        "pages" => Dict(id => read_page(p) for (id, p) in state.pages),
        "epochs" => Dict(id => p.metadata[:epoch_dirty] for (id, p) in state.pages),
        "graph" => serialize_graph(state.graph),
        "next_page_id" => state.next_page_id[],
        "next_delta_id" => state.next_delta_id[],
    )
end

@testset "Pooled state determinism" begin
    fresh = MMSBState()
    pooled = get_pooled_state!()
    
    for state in [fresh, pooled]
        page = API.create_page(state; size=1024)
        API.update_page(state, page.id, rand(UInt8, 1024))
    end
    
    @test canonical_snapshot(fresh) == canonical_snapshot(pooled)
end
```
**File:** `test/determinism_test.jl` (new)
**Status:** 🔨 To implement

---

### Level 2 — Safe Performance Unlocks (PHASE 1)

**T2.1 — Cache-Aware Propagation** 
Implemented in T1.2 above.
**Expected:** 169 μs → 90 μs
**Status:** 🔨 Blocked by T1.1, T1.2

**T2.2 — Remove Redundant Diff Computation**

**Solution:**
```julia
function update_page(state, page_id, bytes)
    page = get_page(state, page_id)
    current = read_page(page)
    
    diff_result = compute_diff(current, bytes)
    diff_result.changed || return page
    
    page.metadata[:last_diff] = diff_result
    delta = create_delta(state, page_id, diff_result.mask, bytes)
    route_delta!(state, delta)
    return page
end
```
**Expected:** -5 μs
**Status:** 🔨 To implement

**T2.3 — Atomic Delta ID Allocation**

**Solution:**
```julia
mutable struct MMSBState
    next_delta_id::Threads.Atomic{UInt64}
end

function allocate_delta_id!(state)
    id = Threads.atomic_add!(state.next_delta_id, UInt64(1))
    return DeltaID(id)
end
```
**Expected:** -10 μs (removes lock)
**File:** `src/01_types/MMSBState.jl:111-117`
**Status:** 🔨 To implement

---

### Level 3 — Allocation & Lifecycle Fixes (PHASE 2A)

**T3.1 — Formal State Reset Specification**

State reset must clear:
1. pages::Dict{PageID, Page}
2. graph::ShadowPageGraph
3. next_page_id, next_delta_id (reset to 1)
4. TLog entries
5. Propagation buffers
6. Cached signatures

**Status:** 📝 Document in `docs/state_lifecycle.md`

**T3.2 — State Reset Implementation**

**Solution:**
```julia
function reset!(state::MMSBState)
    empty!(state.pages)
    state.graph = ShadowPageGraph()
    state.next_page_id[] = PageID(1)
    Threads.atomic_store!(state.next_delta_id, UInt64(1))
    FFIWrapper.rust_tlog_clear!(state.tlog_handle)
    if haskey(PropagationEngine.PROPAGATION_BUFFERS, state)
        delete!(PropagationEngine.PROPAGATION_BUFFERS, state)
    end
end
```
**File:** `src/01_types/MMSBState.jl`
**Status:** 🔨 After T3.1

**T3.3 — State Pool Determinism Test**

Implemented in T1.3 + checkpoint replay test.
**Status:** 🔨 Blocked by T3.2

**T3.4 — State Pool Integration**

**Solution:**
```julia
const STATE_POOL = Channel{MMSBState}(10)

function get_pooled_state!(config)
    if isready(STATE_POOL)
        state = take!(STATE_POOL)
        reset!(state)
        state.config = config
        return state
    end
    return MMSBState(config)
end

function return_to_pool!(state)
    isopen(STATE_POOL) && put!(STATE_POOL, state)
end
```
**Expected:** Allocation 6 μs → 2-3 μs
**File:** `src/01_types/MMSBState.jl`
**Status:** 🔨 Blocked by T3.3

---

### Level 4 — Event & ID Scaling (PHASE 2B)

**T4.1 — Batch Event Emission API**

**Solution:**
```julia
struct EventBatch
    events::Vector{Tuple{EventType, PageID, Any...}}
end

function collect_events!(state)
    batch = EventBatch(Vector{Tuple}())
    state.metadata[:event_batch] = batch
    return batch
end

function emit_event!(state, event_type, args...)
    batch = get(state.metadata, :event_batch, nothing)
    if batch !== nothing
        push!(batch.events, (event_type, args...))
    else
        emit_event_immediate!(state, event_type, args...)
    end
end

function flush_events!(state, batch)
    for (event_type, args...) in batch.events
        emit_event_immediate!(state, event_type, args...)
    end
    empty!(batch.events)
end
```
**Expected:** -5 μs
**File:** `src/03_events/EventSystem.jl`
**Status:** 🔨 To implement

**T4.2 — Event Ordering Contract**

Events within propagation cycle:
1. Topological order of affected pages
2. Within same page: delta epoch order
3. Event type order: INVALIDATED → APPLIED → CHANGED

**Status:** 📝 Document in `docs/event_semantics.md`

**T4.3 — Delta ID Range Allocator**

**Solution:**
```julia
const RANGE_SIZE = 1000

struct DeltaIDAllocator
    ranges::Dict{Int, Vector{UInt64}}
    global_counter::Threads.Atomic{UInt64}
    lock::ReentrantLock
end

function allocate_delta_id!(alloc)
    tid = Threads.threadid()
    range = lock(alloc.lock) do
        get!(alloc.ranges, tid, UInt64[])
    end
    
    if isempty(range)
        start = Threads.atomic_add!(alloc.global_counter, UInt64(RANGE_SIZE))
        append!(range, start:(start+RANGE_SIZE-1))
    end
    
    return DeltaID(pop!(range))
end
```
**Expected:** Better scaling >8 threads
**Status:** 🔨 Optional if T2.3 sufficient

---

### Level 5 — Batch & Parallelism (PHASE 2C)

**T5.1 — Deterministic Batch Routing Order**

**Solution:**
```julia
function batch_route_deltas!(state, deltas)
    grouped = Dict{PageID, Vector{Delta}}()
    for delta in deltas
        push!(get!(grouped, delta.page_id, Delta[]), delta)
    end
    
    page_ids = sort(collect(keys(grouped)))  # Deterministic
    
    for page_id in page_ids
        delta_group = grouped[page_id]
        sorted = sort(delta_group; by = d -> d.epoch)
        for delta in sorted
            route_delta!(state, delta; propagate=false)
        end
    end
    
    propagate_change!(state, page_ids)
end
```
**File:** `src/02_semiring/DeltaRouter.jl:71-87`
**Status:** ✅ Mostly implemented, needs explicit sort

**T5.2 — Parallel Batch Routing**

**Solution:**
```julia
function batch_route_deltas!(state, deltas)
    grouped = Dict{PageID, Vector{Delta}}()
    # ... grouping ...
    page_ids = sort(collect(keys(grouped)))
    
    Threads.@threads for page_id in page_ids
        delta_group = grouped[page_id]
        sorted = sort(delta_group; by = d -> d.epoch)
        for delta in sorted
            route_delta!(state, delta; propagate=false)
        end
    end
    
    propagate_change!(state, page_ids)
end
```
**Expected:** Batch 32×4: 4.0 ms → 3.0 ms
**File:** `src/02_semiring/DeltaRouter.jl`
**Status:** 🔨 To implement

---

### Level 6 — GPU Isolation & Routing (PHASE 2D)

**T6.1 — CPU Fast Path Isolation**

**Solution:**
```julia
function route_delta!(state, delta; propagate=true)
    page = get_page(state, delta.page_id)
    
    if page.location == CPU_LOCATION
        @inbounds FFIWrapper.rust_delta_apply_cpu!(page.handle, delta.handle)
    else
        FFIWrapper.rust_delta_apply_gpu!(page.handle, delta.handle)
    end
    
    append_to_log!(state, delta)
    track_delta_latency!(state, time_ns() - start_ns)
    propagate && propagate_change!(state, delta.page_id)
end
```
**Audit:** Check all FFI calls in `src/FFIWrapper.jl`
**File:** `src/02_semiring/DeltaRouter.jl:29`
**Status:** 🔍 Needs audit

**T6.2 — GPU Size Threshold Routing**

**Solution:**
```julia
const GPU_SIZE_THRESHOLD = 1024 * 1024  # 1 MB

function should_use_gpu(state, page)
    return state.config.enable_gpu &&
           page.size >= GPU_SIZE_THRESHOLD &&
           page.location == GPU_LOCATION
end
```
**Expected:** Avoid 174× slowdown for small pages
**File:** `src/01_allocator/PageAllocator.jl`
**Status:** 🔨 To implement

**T6.3 — GPU Sync Audit**

Audit FFI for implicit cudaDeviceSynchronize() calls:
- rust_delta_apply_gpu
- rust_page_read
- rust_page_write

Add explicit sync only at propagation boundaries.
**Status:** 🔍 Needs Rust-side audit

---

### Level 7 — Validation & Benchmark Closure (PHASE 3)

**T7.1 — Benchmark Re-run (Phase 1)**

Command: `cd benchmark && julia benchmarks.jl`
Success: `propagation/single_hop` < 60 μs
**Status:** ⏳ After T2.1, T2.2, T2.3

**T7.2 — Benchmark Re-run (Phase 2)**

Success criteria:
- `allocation/cpu_1kb` < 3 μs
- `propagation/single_hop` < 45 μs
**Status:** ⏳ After T3.4, T4.3, T5.2, T6.3

**T7.3 — Target Reclassification**

Update documentation:
```markdown
## Performance Targets

### Architectural (Phase 2)
- Allocation: 2 μs
- Propagation: 40 μs

### Ultimate (Future)
- Allocation: 1 μs
- Propagation: 10 μs
```
**File:** `README.md`
**Status:** 📝 Document

---

## Status Legend

- ✅ Already correct
- 📝 Document needed
- 🔍 Needs audit
- 🔨 To implement
- ⏳ Blocked
- ⚠️ Needs decision

**Next action:** Start Level 0 documentation before implementation.
