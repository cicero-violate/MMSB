# Benchmark Implementation DAG & Task Log

**Date:** 2025-12-17  
**Status:** Phase 6 Planning

## Overview

Phases 1-5 complete ✓. Phase 6 implements benchmark validation infrastructure.

---

## Phase 6: Benchmark Validation (Weeks 27-33)

### Dependencies

```
Critical Features (P0, Week 27-28)
    ├─> Ring Buffer
    └─> Columnar Storage
        |
        v
Correctness Infrastructure (P1, Week 29-30)
    ├─> Replay Validator (#1)
    ├─> Integrity Checker (#2)
    ├─> Graph Validator (#3)
    └─> Invariant Framework (#8)
        |
        v
Performance Engines (P1, Week 31-32)
    ├─> Throughput Engine (#5)
    ├─> Tick Orchestrator (#6)
    └─> Memory Monitor (#7)
        |
        v
Testing & Validation (P2, Week 33)
    ├─> Purity Validator (#4)
    ├─> Stability Tests (#9)
    ├─> Provenance Tracker (#10)
    └─> Benchmark Suite (tests/benchmark_*.rs)
```

---

## Task Breakdown

### Week 27-28: Critical Infrastructure (P0)

#### T6.1: Lock-Free Ring Buffer
**File:** `src/04_propagation/ring_buffer.rs`
**Blocks:** #5 (Throughput), #6 (Tick Latency)
**Deps:** None
**Est:** 2 days

Tasks:
- [x] Implement `LockFreeRingBuffer<T>` with AtomicUsize head/tail
- [x] Cache-line alignment (64-byte boundaries)
- [x] Try-push/try-pop with CAS operations
- [x] Batch enqueue/dequeue methods
- [x] Unit tests: concurrent access, wraparound, full/empty states

Validation:
- Benchmark vs VecDeque: expect 3-5x lower latency
- Zero allocations after initialization

---

#### T6.2: Columnar Delta Storage
**File:** `src/01_page/columnar_delta.rs`
**Blocks:** #5 (Throughput), #7 (Memory)
**Deps:** None
**Est:** 3 days

Tasks:
- [x] `ColumnarDeltaBatch` struct (SOA layout)
- [x] Conversion from/to `Vec<Delta>`
- [x] SIMD field filtering (epoch, page_id scans)
- [x] Batch apply to pages
- [x] Integration with `ThroughputEngine`

Validation:
- Cache miss rate: expect 50% reduction vs row layout
- SIMD throughput: 4-8x speedup on AVX2/AVX512

---

### Week 29-30: Correctness (P1)

#### T6.3: Replay Validator (#1)
**File:** `src/01_page/replay_validator.rs`
**Blocks:** Benchmark #1
**Deps:** TLog infrastructure
**Est:** 2 days

Tasks:
- [x] `ReplayValidator` struct with checkpoint storage
- [x] State snapshot/restore integration
- [x] L2 divergence computation
- [x] Test: 1000 cycles, verify divergence < 1e-9

---

#### T6.4: Delta Integrity Checker (#2)
**File:** `src/01_page/integrity_checker.rs`
**Blocks:** Benchmark #2
**Deps:** `DeviceBufferRegistry`
**Est:** 1 day

Tasks:
- [x] Schema validation (mask/payload alignment)
- [x] Orphan delta detection (missing PageID)
- [x] Epoch monotonicity checks
- [x] Test: 1M deltas, verify 0 violations

---

#### T6.5: Graph Cycle Detection (#3)
**File:** `src/03_dag/graph_validator.rs`
**Blocks:** Benchmark #3
**Deps:** `ShadowPageGraph`
**Est:** 2 days

Tasks:
- [x] Tarjan's algorithm for cycle detection
- [x] Per-page validation (<1ms target)
- [x] Early termination on first cycle
- [x] Test: 10k pages with various topologies

---

#### T6.6: Invariant Framework (#8)
**File:** `src/06_utility/invariant_checker.rs`
**Blocks:** Benchmark #8
**Deps:** None
**Est:** 2 days

Tasks:
- [x] `Invariant` trait definition
- [x] Built-in invariants: EpochMonotonicity, PageConsistency, GraphAcyclicity
- [x] Batch validation of all invariants
- [x] Test: 10k cycles, verify 0 failures

---

### Week 31-32: Performance (P1)

#### T6.7: Throughput Engine (#5)
**File:** `src/04_propagation/throughput_engine.rs`
**Blocks:** Benchmark #5
**Deps:** T6.1 (Ring Buffer), T6.2 (Columnar)
**Est:** 3 days

Tasks:
- [x] Thread pool with work-stealing
- [x] Partition deltas by PageID
- [x] Integrate columnar batching
- [x] SIMD merge from `delta_merge.rs`
- [x] Test: 10M deltas/sec multi-threaded

---

#### T6.8: Tick Orchestrator (#6)
**File:** `src/04_propagation/tick_orchestrator.rs`
**Blocks:** Benchmark #6
**Deps:** T6.7, T6.9
**Est:** 2 days

Tasks:
- [x] Propagation phase (<10ms budget)
- [x] Graph resolution phase (<4ms budget)
- [x] Conditional GC phase (<2ms budget)
- [x] High-resolution timer instrumentation
- [x] Test: sustained <16ms ticks

---

#### T6.9: Memory Monitor (#7)
**File:** `src/06_utility/memory_monitor.rs`
**Blocks:** Benchmark #7
**Deps:** `AllocatorStats`
**Est:** 2 days

Tasks:
- [x] Real-time memory snapshot
- [x] Incremental GC trigger
- [x] Page aging heuristics
- [x] Test: 1M pages ≤ 1GB RAM, GC < 3ms

---

### Week 33: Testing & Validation (P2)

#### T6.10: Purity Validator (#4)
**File:** `src/02_semiring/purity_validator.rs`
**Blocks:** Benchmark #4
**Deps:** None
**Est:** 1 day

Tasks:
- [x] Referential transparency checks
- [x] Multiple-run output comparison
- [x] Test: validate all semiring operations

---

#### T6.11: Stability Tests (#9)
**File:** `tests/benchmark_09_stability.rs`
**Blocks:** Benchmark #9
**Deps:** None
**Est:** 1 day

Tasks:
- [x] Perturbation injection (Gaussian noise)
- [x] State divergence tracking
- [x] NaN/Inf detection
- [x] Trajectory smoothness metrics

---

#### T6.12: Provenance Tracker (#10)
**File:** `src/06_utility/provenance_tracker.rs`
**Blocks:** Benchmark #10
**Deps:** `ShadowPageGraph`
**Est:** 1 day

Tasks:
- [x] Backward graph traversal
- [x] LRU cache for resolved chains
- [x] Depth limit enforcement (<32)
- [x] Test: resolution < 50ms

---

#### T6.13: Benchmark Test Suite
**Files:** `tests/benchmark_{01..10}_*.rs`
**Blocks:** All benchmarks
**Deps:** T6.3-T6.12
**Est:** 2 days

Tasks:
- [x] Create 10 benchmark test files
- [x] Integrate with `cargo test --release`
- [x] CI pipeline integration
- [x] Capture baseline results in `benchmark/results/`

---

## Task Log

### 2025-12-17
- [x] Phase 6 planning complete
- [x] Feature gap analysis (FEATURE_ANALYSIS.md)
- [x] Updated PROPOSAL.md with ring buffer & columnar storage
- [x] Begin T6.1: Ring buffer implementation

---

## Success Metrics

Phase 6 complete when:
```bash
cargo test --release --test benchmark_01_replay       # PASS
cargo test --release --test benchmark_02_integrity    # PASS
cargo test --release --test benchmark_03_graph        # PASS
cargo test --release --test benchmark_04_isolation    # PASS
cargo test --release --test benchmark_05_throughput   # PASS (≥10M/sec)
cargo test --release --test benchmark_06_tick_latency # PASS (<16ms)
cargo test --release --test benchmark_07_memory       # PASS (≤1KB/page)
cargo test --release --test benchmark_08_invariants   # PASS (0 violations)
cargo test --release --test benchmark_09_stability    # PASS (no NaNs)
cargo test --release --test benchmark_10_provenance   # PASS (<50ms)
```

---

## Phases 7-8 Preview

### Phase 7: Observability (Week 34)
- Prometheus exporter
- Flamegraph integration
- Trace visualization
- Regression test CI

### Phase 8: Documentation & Examples (Week 35-36)
- Complete API reference (Layers 0-12)
- Example applications (compiler IR, game AI, finance)
- Production deployment guide

**END DAG**
