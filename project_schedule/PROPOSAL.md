# MMSB Benchmark Implementation Proposal

**Date:** 2025-12-17  
**Status:** Draft v2 (Updated with missing features)

## Executive Summary

Analysis reveals MMSB has strong Layer 0-6 foundations but requires:
1. **Missing critical features:** Ring buffer, columnar storage
2. **Validation infrastructure:** Replay validation, graph cycle detection, invariant checkers
3. **Performance optimization:** Throughput engines, tick orchestration, GC integration

---

## Feature Gap Analysis

### Implemented
- Lock-free allocator (partial: ≤4KB pages only)
- Batch delta operations (partial: queue only, no SIMD apply)
- Zero-copy access (partial: internal only, FFI copies)
- Page/Delta/TLog/Checkpoint infrastructure
- SIMD delta merge (AVX2/AVX512)
- GPU memory pools, NCCL integration

### Missing (Blocks Benchmarks)
- **Ring buffer** (using VecDeque, reallocates) → Affects #5, #6
- **Columnar storage** (row-oriented) → Affects #5, #7
- Deterministic replay validator → Benchmark #1
- Graph cycle detection → Benchmark #3
- Purity validators → Benchmark #4
- Throughput engine → Benchmark #5
- Tick orchestrator → Benchmark #6
- Memory monitor with GC → Benchmark #7
- Invariant framework → Benchmark #8
- Stability tests → Benchmark #9
- Provenance tracker → Benchmark #10

---

## Implementation Plan

### Phase 1: Critical Infrastructure (Week 1-2)

#### 1.1 Lock-Free Ring Buffer
**File:** `src/04_propagation/ring_buffer.rs`

Variables:
- $R[N]$ = fixed-size circular buffer
- $h, t \in [0, N)$ = head, tail indices
- $(t + 1) \mod N = h \implies$ full

```rust
pub struct LockFreeRingBuffer<T> {
    buffer: Box<[MaybeUninit<T>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
}
```

Target: O(1) enqueue/dequeue, zero allocations, cache-aligned

#### 1.2 Columnar Delta Storage
**File:** `src/01_page/columnar_delta.rs`

Variables:
- $\Delta_{\text{col}} = \{[id_1, \ldots], [epoch_1, \ldots], [mask_1, \ldots]\}$
- Cache lines: $\lceil n \cdot \text{sizeof}(field) / 64 \rceil < \lceil n \cdot \text{sizeof}(Delta) / 64 \rceil$

```rust
pub struct ColumnarDeltaBatch {
    count: usize,
    delta_ids: Vec<u64>,      // SOA layout
    page_ids: Vec<u64>,
    epochs: Vec<u32>,
    mask_pool: Vec<u8>,
    payload_pool: Vec<u8>,
}
```

Explanation: Structure-of-Arrays enables SIMD processing of individual fields without loading entire structs. Filtering by epoch scans contiguous u32 array vs scattered struct fields.

Target: 10x better cache efficiency for selective field access

#### 1.3 Replay Validator
**File:** `src/01_page/replay_validator.rs`

Variables:
- $S_t$ = state at tick $t$
- $\|S_{\text{orig}}(t) - S_{\text{replay}}(t)\|_2 < 10^{-9}$

```rust
pub struct ReplayValidator {
    checkpoints: Vec<StateSnapshot>,
    threshold: f64,
}

impl ReplayValidator {
    pub fn validate(&self, tlog: &TransactionLog) -> ReplayReport;
}
```

Target: Divergence < 1e-9 over 1000 cycles

#### 1.4 Graph Cycle Detection
**File:** `src/03_dag/graph_validator.rs`

Variables:
- $G = (V, E)$ where $\nexists$ path $v \to v$
- $T_{\text{detect}} = O(V + E)$ via Tarjan's algorithm

```rust
pub struct GraphValidator;

impl GraphValidator {
    pub fn detect_cycles(&self, graph: &ShadowPageGraph) -> CycleReport;
    pub fn validate_page(&self, page_id: PageID) -> ValidationResult;
}
```

Target: <10ms for 10k pages, <1ms per page validation

---

### Phase 2: Performance Engines (Week 3-4)

#### 2.1 Throughput Engine
**File:** `src/04_propagation/throughput_engine.rs`

Variables:
- $T = N_{\text{deltas}} / t_{\text{elapsed}}$
- Target: $T_1 \geq 10^6$/sec single, $T_N \geq 10^7$/sec multi

```rust
pub struct ThroughputEngine {
    workers: ThreadPool,
    batch_size: usize,
}

impl ThroughputEngine {
    pub fn process_parallel(&self, deltas: Vec<Delta>) -> Metrics;
}
```

Optimizations:
- Partition by PageID
- SIMD merge from `delta_merge.rs`
- Lock-free ring buffer
- Columnar batch processing

#### 2.2 Tick Orchestrator
**File:** `src/04_propagation/tick_orchestrator.rs`

Variables:
- $T_{\text{tick}} = T_{\text{prop}} + T_{\text{resolve}} + T_{\text{gc}} < 16\text{ms}$

```rust
pub struct TickOrchestrator {
    propagation: PropagationEngine,
    graph: Arc<ShadowPageGraph>,
}

impl TickOrchestrator {
    pub fn execute_tick(&mut self) -> TickMetrics;
}
```

Target: <16ms for 62.5Hz simulation

#### 2.3 Memory Monitor
**File:** `src/06_utility/memory_monitor.rs`

Variables:
- $M_{\text{avg}} = M_{\text{total}} / N \leq 1024$ bytes
- $T_{\text{GC}} < 3\text{ms}$

```rust
pub struct MemoryMonitor {
    stats: Arc<AllocatorStats>,
    gc_threshold: usize,
}

impl MemoryMonitor {
    pub fn snapshot(&self) -> MemorySnapshot;
    pub fn trigger_gc(&self) -> Option<GCMetrics>;
}
```

Target: 1M pages ≤ 1GB RAM, GC < 3ms

---

### Phase 3: Validation Framework (Week 5)

#### 3.1 Invariant Checker
**File:** `src/06_utility/invariant_checker.rs`

Variables:
- $\forall t: \text{Invariants}(S_t) \implies \text{Invariants}(S_{t+1})$

```rust
pub trait Invariant {
    fn check(&self, state: &MMSBState) -> InvariantResult;
}

struct EpochMonotonicity;
struct PageConsistency;
struct GraphAcyclicity;
```

Target: 0 failures over 10k cycles

#### 3.2 Purity Validator
**File:** `src/02_semiring/purity_validator.rs`

```rust
pub struct PurityValidator;

impl PurityValidator {
    pub fn validate<F, I, O>(&self, func: F, inputs: Vec<I>) -> Report
    where F: Fn(I) -> O;
}
```

Target: 100% purity for all algorithms

#### 3.3 Stability Tests
**File:** `tests/benchmark_09_stability.rs`

Variables:
- $\tilde{\delta} = \delta + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- $\|\tilde{S}_t - S_t\| \leq K \cdot \|\epsilon\|$

Target: Bounded error, no NaNs, smooth trajectories

#### 3.4 Provenance Tracker
**File:** `src/06_utility/provenance_tracker.rs`

Variables:
- $\text{Chain}(\delta_n) = \{\delta_0, \ldots, \delta_n\}$
- $T_{\text{resolve}} < 50\text{ms}$, depth < 32

```rust
pub struct ProvenanceTracker {
    graph: Arc<ShadowPageGraph>,
    cache: LruCache<DeltaID, Chain>,
}
```

---

### Phase 4: Integration & Benchmarking (Week 6-7)

#### 4.1 Benchmark Test Suite
Create `tests/benchmark_{01..10}_*.rs`:
1. `replay.rs` - Deterministic replay
2. `integrity.rs` - Delta schema validation
3. `graph.rs` - Cycle detection
4. `isolation.rs` - Algorithm purity
5. `throughput.rs` - Delta processing rate
6. `tick_latency.rs` - Simulation tick time
7. `memory.rs` - Footprint & GC
8. `invariants.rs` - Constraint preservation
9. `stability.rs` - Perturbation response
10. `provenance.rs` - Traceability

#### 4.2 Integration Points
- Replace `PropagationQueue::VecDeque` with `LockFreeRingBuffer`
- Add `ColumnarDeltaBatch` fast path in throughput engine
- Wire memory monitor into allocator
- Register invariants with orchestrator

---

## Success Criteria

All benchmarks pass:
```
✓ #1:  Replay (divergence: <1e-9)
✓ #2:  Integrity (orphans: 0)
✓ #3:  Graph (cycle-detect: <10ms/10k)
✓ #4:  Isolation (purity: 100%)
✓ #5:  Throughput (≥10M/sec)
✓ #6:  Tick Latency (<16ms)
✓ #7:  Memory (avg: ≤1KB/page, GC: <3ms)
✓ #8:  Invariants (failures: 0/10k)
✓ #9:  Stability (no NaNs, bounded)
✓ #10: Provenance (<50ms, depth <32)
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ring buffer lock contention | High | Use separate rings per thread |
| Columnar conversion overhead | Medium | Lazy conversion, hybrid approach |
| GC pauses exceed 3ms | High | Incremental GC, page aging |
| SIMD unavailable | Medium | Scalar fallback paths |

---

## File Structure

New files:
```
src/
├── 01_page/
│   ├── columnar_delta.rs          ← NEW (columnar storage)
│   ├── replay_validator.rs        ← NEW (#1)
│   └── integrity_checker.rs       ← NEW (#2)
├── 02_semiring/
│   └── purity_validator.rs        ← NEW (#4)
├── 03_dag/
│   └── graph_validator.rs         ← NEW (#3)
├── 04_propagation/
│   ├── ring_buffer.rs             ← NEW (ring buffer)
│   ├── throughput_engine.rs       ← NEW (#5)
│   └── tick_orchestrator.rs       ← NEW (#6)
└── 06_utility/
    ├── memory_monitor.rs          ← NEW (#7)
    ├── invariant_checker.rs       ← NEW (#8)
    └── provenance_tracker.rs      ← NEW (#10)

tests/
└── benchmark_{01..10}_*.rs        ← NEW (all 10)
```

**END PROPOSAL**
