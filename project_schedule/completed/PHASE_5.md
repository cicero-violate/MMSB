# Phase 5: Production Hardening - DAG & Priorities

## Dependency Graph

```
PHASE4-COMPLETE ── BENCHMARKS ── GPU-OPT ── PERF-OPT ── RELIABILITY ── OBSERVABILITY ── PHASE5-COMPLETE
                       │            │          │             │              │
                       │            └──────────┴─────────────┴──────────────┘
                       │                                     │
                       └─────────────────────────────────────┘
```

## Priority Task List

### P0: Critical Path

| Task ID | Task                                    | Deps         | Owner | Status |
|---------|-----------------------------------------|--------------|-------|--------|
| B.1     | Core propagation benchmark suite        | PHASE4       | -     | ☐      |
| B.2     | Allocator performance profiling         | PHASE4       | -     | ☐      |
| B.3     | End-to-end pipeline benchmarks          | B.1, B.2     | -     | ☐      |
| G.1     | Persistent CUDA kernel for propagation  | B.1          | -     | ☐      |
| G.2     | GPU memory pool implementation          | G.1          | -     | ☐      |
| P.1     | SIMD delta merging                      | B.2          | -     | ☐      |
| P.2     | Lock-free allocation fast path          | B.2          | -     | ☐      |
| P.3     | Zero-copy FFI for large transfers       | B.3          | -     | ☐      |
| R.1     | Error recovery framework                | PHASE4       | -     | ☐      |
| R.2     | Graceful GPU fallback                   | G.2          | -     | ☐      |
| O.1     | Prometheus metrics exporter             | PHASE4       | -     | ☐      |
| O.2     | Performance regression test suite       | B.3          | -     | ☐      |

### P1: High Priority

| Task ID | Task                              | Deps   | Owner | Status |
|---------|-----------------------------------|--------|-------|--------|
| G.3     | Multi-GPU support (NCCL)          | G.2    | -     | ☐      |
| G.4     | Unified memory prefetching        | G.2    | -     | ☐      |
| G.5     | CUDA graph capture                | G.1    | -     | ☐      |
| P.4     | Compressed delta representation   | P.1    | -     | ☐      |
| P.5     | Batch propagation API             | P.3    | -     | ☐      |
| R.3     | Memory pressure handling          | R.1    | -     | ☐      |
| R.4     | Checkpoint corruption detection   | R.1    | -     | ☐      |
| R.5     | Transaction isolation guarantees  | R.4    | -     | ☐      |
| O.3     | Flamegraph profiling integration  | O.1    | -     | ☐      |
| O.4     | Memory usage heatmaps             | O.1    | -     | ☐      |
| O.5     | Propagation trace visualization   | O.2    | -     | ☐      |

### P2: Nice to Have

| Task ID | Task                         | Deps | Owner | Status |
|---------|------------------------------|------|-------|--------|
| D.1     | Complete Layer 0-4 docs      | -    | -     | ☐      |
| D.2     | Complete Layer 5-7 docs      | -    | -     | ☐      |
| D.3     | Complete Layer 8-9 docs      | -    | -     | ☐      |
| D.4     | API reference documentation  | -    | -     | ☐      |
| E.1     | Example: compiler IR         | -    | -     | ☐      |
| E.2     | Example: game AI             | -    | -     | ☐      |
| E.3     | Example: financial modeling  | -    | -     | ☐      |

## Success Criteria

- [ ] 10x improvement in hot path latency
- [ ] <5% overhead vs raw CUDA
- [ ] Zero test failures under memory pressure
- [ ] Metrics cover all critical paths
- [ ] Graceful degradation without GPU

## Timeline Estimate

- Benchmarking: 1 week
- GPU Optimization: 2 weeks
- Performance: 2 weeks
- Reliability: 1 week
- Observability: 1 week

**Total: ~7 weeks**
