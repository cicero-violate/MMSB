# Phase 6 DAG & Dependencies

**Status**: Phases 1-5 Complete ✓

## DAG Structure

```
Benchmarking → GPU/Performance → Reliability → Observability → Documentation → Examples
```

## Tasks

### Benchmarking (P1) - Week 27
- [x] Allocator performance
  - Capture CPU/GPU allocation latency with the `SUITE["allocation"]` group in `benchmark/benchmarks.jl`, profiling the hot paths implemented in `src/00_physical/allocator.rs`.
  - Export medians/min/max to `benchmark/results/baseline.json` so week-over-week regressions can be compared with `benchmark/helpers.jl`.
- [x] Semiring operations
  - Add microbenchmarks that call `src/02_semiring/semiring_ops.rs` (`fold_add`, `fold_mul`, `accumulate`) and their Julia wrappers in `src/02_semiring/Semiring.jl`.
  - Record per-semiring throughput (tropical/boolean/custom) so optimizations inside `src/02_semiring/DeltaRouter.jl` have a measurable target.
- [x] Graph traversal
  - Exercise traversal helpers such as `src/03_dag/shadow_graph_traversal.rs` and the Julia ShadowPageGraph utilities to quantify topological-sort and BFS costs.
  - Feed synthetic DAGs from `src/05_graph` through the benchmarks to stress dependency fan-out/fan-in shapes observed in production traces.
- [x] Propagation performance
  - Benchmark the fast path defined in `src/04_propagation/propagation_fastpath.rs` and the scheduler in `src/04_propagation/PropagationEngine.jl` with varying batch sizes.
  - Use instrumentation from `src/04_instrumentation` so allocator, semiring, and propagation timing can be correlated inside a single JSON report.
- [x] Full system benchmarks
  - Reuse the stress scenario at the bottom of `benchmark/benchmarks.jl` and extend it with multi-page DAGs plus checkpoint/replay loops from `src/01_page/tlog.rs`.
  - Summarize allocator + semiring + propagation latency in a combined report so Week 28 optimization targets are grounded in measured data.

### GPU Optimization (P1) - Weeks 28-29
- [x] Persistent kernels
  - Extend `src/04_propagation/gpu_propagation.cu` so a resident kernel consumes command buffers emitted by `propagation_command_buffer.rs` without per-delta launches.
  - Manage the GPU work queue from `src/00_physical/DeviceSync.jl`, ensuring host/device signaling stats are visible via the instrumentation hooks.
- [x] GPU memory pool
  - Implement a slab allocator in `src/00_physical/UnifiedMemory.jl` (and mirror logic in `allocator.rs`) so repeated page allocations reuse CUDA buffers.
  - Surface pool metrics through the allocator stats module and expose a Julia knob in `src/00_physical/PageAllocator.jl`.
- [x] Multi-GPU NCCL
  - Integrate NCCL bindings alongside the device registry in `src/03_device/device_registry.rs` to negotiate communicators for every active GPU.
  - Add a collective delta path in `src/04_propagation/gpu_propagation.cu` so all-reduce/all-gather synchronizes propagation buffers across devices.
- [x] Prefetch tuning
  - Use the migration hooks already in `src/00_physical/UnifiedMemory.jl` to issue `cudaMemPrefetchAsync` before large propagations.
  - Feed telemetry back into the `src/06_utility` cost functions so prefetch distance adapts to observed latency.
- [x] CUDA graph capture
  - Capture the steady-state propagation sequence (command buffer build → persistent kernel wakeup) with CUDA Graph APIs inside `gpu_propagation.cu`.
  - Provide a Julia toggle in `src/04_propagation/PropagationEngine.jl` so graph replay can be switched on for workloads with repetitive structures.

### Performance (P1) - Weeks 30-31
- [x] SIMD delta merge
  - Specialize `src/01_page/delta_merge.rs` with AVX2/AVX-512 intrinsics and export CPU feature detection through `src/06_utility/cpu_features.rs`.
  - Mirror the optimized path in Julia via `src/01_page/Delta.jl` for environments that JIT hot loops instead of using the Rust kernels.
- [x] Lock-free allocator
  - Introduce an atomic freelist for small pages inside `src/00_physical/allocator.rs`, guarded by the fast-path helpers already used in `PageAllocator`.
  - Retain the Mutex-backed slow path for large buffers while surfacing contention counters in `src/00_physical/allocator_stats.rs`.
- [x] Zero-copy FFI
  - Audit the pointer conversions in `src/ffi.rs` and `src/ffi/ffi.rs` (Julia side) to ensure deltas/pages can be shared without memcpy.
  - Annotate ownership rules in `docs/API.md` so embedders know when they must copy vs borrow memory.
- [x] Delta compression
  - Extend `src/01_page/tlog_compression.rs` with sparsity-aware encoders (RLE/bitpack) and toggle them from `src/01_page/TLog.jl`.
  - Persist compression ratios inside the log summary so checkpoint size regressions are visible to the benchmarking suite.
- [x] Batch propagation
  - Finish wiring `src/04_propagation/propagation_queue.rs` so batched deltas flush through a single notify in `PropagationEngine.jl`.
  - Expose a Julia API (`batch_route_deltas!` in `docs/API.md`) that mirrors the lower-level queue to amortize synchronization.

### Reliability (P1) - Week 32
- [x] Error recovery
  - Formalize an error taxonomy shared between Rust and Julia (`src/ffi.rs` ↔ `src/API.jl`) and centralize retry/backoff logic in `src/06_utility/ErrorRecovery.jl`.
  - Route fatal errors through the observability hooks so production deployments capture context before retrying.
- [x] GPU fallback
  - Use the capability probes in `src/00_physical/DeviceSync.jl` to auto-disable GPU paths when CUDA initialization fails mid-flight.
  - Provide CPU equivalents for persistent kernels/queues so `PropagationEngine.jl` can continue processing without user intervention.
- [x] Memory pressure handling
  - Teach `src/06_utility/MemoryPressure.jl` (new) to read allocator stats and start evicting pages (LRU) through the page manager in `src/01_page/page.rs`.
  - Tie pressure events to the checkpoint subsystem so cold pages spill to disk via the existing `src/01_page/checkpoint.rs`.
- [x] Checkpoint validation
  - Embed CRC32/SHA256 checks in `src/01_page/checkpoint.rs` plus verification in the replay path before data touches live pages.
  - Surface validation failures as structured errors that `docs/API.md` can document for operators.
- [x] Transaction isolation
  - Implement per-transaction epochs inside `src/04_propagation/PropagationEngine.jl` so overlapping intents do not step on each other's state.
  - Add Julia helpers (e.g., `with_transaction`) in `src/10_agent_interface/CheckpointAPI.jl` ensuring agents pick the right isolation level.

### Observability (P1) - Week 33
- [ ] Prometheus exporter
- [ ] Regression test CI
- [ ] Flamegraph integration
- [ ] Memory heatmaps
- [ ] Trace visualization

### Documentation (P1) - Weeks 34-35
- [ ] Layer 0-4 API docs
- [ ] Layer 5-7 API docs
- [ ] Layer 8-9 API docs
- [ ] Complete API reference

### Examples (P2) - Week 36
- [ ] Compiler IR example
- [ ] Game AI example
- [ ] Finance example
