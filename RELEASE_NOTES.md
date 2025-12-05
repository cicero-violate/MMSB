# MMSB v0.1.0-alpha Release Notes

**Tag:** `v0.1.0-alpha`  
**Date:** 2025-12-04  
**Status:** Validated for public preview (docs complete, tests passing, examples runnable).

## Highlights
- Documentation set (`README.md`, `docs/*`, `examples/*`) now covers architecture, API, serialization, and runnable workflows.
- Core runtime, GPU kernels, propagation engine, monitoring stack, and benchmark harness are production ready.
- Metadata and dependency tweaks ensure `Pkg.test()` works out of the box with Julia 1.12 (adds `Test` extra).

## Validation Matrix
| Check                    | Command                                                        | Result |
| ---                      | ---                                                            | ---    |
| Unit & integration tests | `julia --startup-file=no --project -e 'using Pkg; Pkg.test()'` | PASS   |
| Quickstart example       | `julia --startup-file=no --project examples/quickstart.jl`     | PASS   |
| Tutorial pipeline        | `julia --startup-file=no --project examples/tutorial.jl`       | PASS   |

## Known Follow-ups
- Performance hot spots (propagation batching, sparse delta SIMD, allocator fast-path) remain Phase 2 work.
- Future documentation expansions: PropagationEngine deep dive, DeviceSync appendix, release automation scripts.

## Phase 2 Progress (post-tag)
- Introduced propagation command buffers and per-state dedup queues to cut invalidation/event overhead.
- Added `route_delta!(; propagate=false)` and upgraded `batch_route_deltas!` so multiple deltas share a single propagation wave.
- Extended the test suite with `Propagation optimization behaviors` validating the new queue semantics and batch routing path.
