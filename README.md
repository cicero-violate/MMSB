# MMSB — Memory-Mapped State Bus

MMSB is a deterministic, delta-driven shared-memory fabric that lets CPU, GPU, and compiler subsystems share page-aligned state under a single transaction log. Every operation follows the MMSB semiring law `state × delta → state′`, so reproducing a run is equivalent to replaying the deltas that touched those pages. This project packages the allocator, delta router, propagation graph, GPU kernels, instrumentation hooks, and monitoring stack needed to operate the bus inside Julia runtimes.

## Why MMSB
- **Deterministic replay** — Byte-level deltas, epochs, and a transaction log (TLog) make it possible to checkpoint or reconstruct any state slice.
- **CPU/GPU coherence** — CPU pages, CUDA buffers, and unified allocations share the same API and propagation semantics.
- **Declarative graph** — ShadowPageGraph captures dependencies explicitly, so propagation is algebraic instead of imperative.
- **Instrumentation hooks** — Julia compiler hooks map SSA/IR artifacts onto pages for debugging and provenance.
- **Observability** — Built-in monitoring records allocator pressure, delta latency, propagation throughput, and log health.

## Quick Start
1. **Install dependencies**
   ```bash
   julia --startup-file=no --project -e 'using Pkg; Pkg.instantiate()'
   ```
2. **Run the quickstart example**
   ```bash
   julia --startup-file=no --project examples/quickstart.jl
   ```
3. **Dig deeper with the tutorial**
   ```bash
   julia --startup-file=no --project examples/tutorial.jl
   ```
4. **Read the docs**
   - `docs/Architecture.md` — layer-by-layer system walkthrough
   - `docs/API.md` — public surface, configuration, and error types
   - `docs/SerializationSpec.md` — binary contract for pages, deltas, and checkpoints

## Repository Map
| Path                     | Purpose                                                               |
| ---                      | ---                                                                   |
| `src/01_types`           | Page, delta, graph, and state definitions (page-aligned state forms). |
| `src/02_runtime`         | Allocator, delta router, TLog, replay engine.                         |
| `src/03_device`          | CUDA kernels plus device sync helpers.                                |
| `src/04_instrumentation` | Compiler hook plumbing and instrumentation manager.                   |
| `src/05_graph`           | Propagation engine and event system.                                  |
| `src/utils`              | Monitoring/statistics helpers.                                        |
| `examples/`              | Runnable Julia demos (`quickstart.jl`, `tutorial.jl`).                |
| `docs/`                  | Architecture, API, and serialization references.                      |
| `benchmark/`             | Benchmark harness and captured baselines.                             |
| `test/`                  | Deterministic regression suites.                                      |

## Release Readiness
- **Current tag:** `v0.1.0-alpha` (see `RELEASE_NOTES.md` for validation + checks).
- **Testing:** `julia --startup-file=no --project -e 'using Pkg; Pkg.test()'`
- **Examples:** `julia --startup-file=no --project examples/quickstart.jl`, `examples/tutorial.jl`
- **Performance Baseline:** Captured in `benchmark/results/baseline.json` with known hotspots (propagation ≈500 μs, sparse delta ≈200 μs, alloc ≈5 μs). Optimization is deferred to the post-release performance track.
- **Phase 2 kickoff:** Propagation command buffers + deduplicated queues reduce redundant recompute work and power `batch_route_deltas!` to update whole waves at once.

## Operational Model
- **Semiring discipline:** Every module treats deltas as additive merges (`⊕`) and propagations as causal applies (`⊗`).
- **Explicit dependencies:** ShadowPageGraph and PropagationEngine are the only coordination backbone; no hidden globals.
- **Command buffers:** All I/O boundaries pass through command buffers/doorbells, so compute paths remain branchless.

## Contributing
1. Run formatting/linting hooks and keep docstrings on every function.
2. Update `docs/*.md` and `examples/*.jl` when changing public behavior.
3. Capture new baselines with `benchmark/benchmarks.jl` when touching performance-sensitive code.
# MMSB
