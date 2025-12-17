# MMSB Production Validation DAG (Active Items Only)

**Date:** 2025-12-17  
**Phase:** 7 – Production Validation (remaining scope)  
**Recent:** Stress suites, Julia harness, and documentation refresh have been archived under `project_schedule/completed/05_DAG_DEPENDENCIES_phase7_completed.md`.

---

## Current Status
- Rust + Julia validation stacks run clean in release (`iterate.sh` updated).
- Throughput/tick benchmarks still below stretch targets (multi-thread + full tick).
- CI/CD automation not yet in place → remaining blocker for closing Phase 7.

---

## Open Tasks

### T7.8 CI/CD Pipeline (P2, 2-3 days)
**Owner:** Platform  
**Files:** `.github/workflows/benchmarks.yml`, `scripts/ci/*`

- [ ] Stand up GitHub Actions workflow (push + weekly schedule).
- [ ] Cache toolchains (Rust stable + Julia 1.10) for faster runs.
- [ ] Run `cargo test --release`, stress suites, and `julia benchmark/validate_all.jl`.
- [ ] Upload benchmark JSON + logs as artifacts for regression tracking.
- [ ] Gate merges on 10/10 validator success (allow override for perf investigations).

### P7-Patch: Performance Gap Closure (P1, parallel)
**Owner:** Runtime/Julia

**Throughput (Rust stress suite)**
- [ ] Capture perf flamegraph for `multi_thread_10m_deltas_per_sec` (`perf record -F 4000`).
- [ ] Batch FFI calls in `DeltaRouter` to reduce lock contention (target: ≤2 syscalls/delta).
- [ ] Enable SIMD merge path by default (guard cleanup + benchmarks).
- [ ] Re-run `tests/stress_throughput.rs` and store metrics in `benchmark/results/phase6.json`.

**Julia Tick Latency**
- [ ] Cache `ensure_rust_artifacts` result in `FFIWrapper.jl` (DONE) and extend cache to `dlopen`.
- [ ] Gate event logging via config (DONE) and verify `state.config.enable_logging=false` everywhere.
- [ ] Profile `_full_system_benchmark!` after changes and shrink tick median <16 ms.
- [ ] Dump new Julia benchmark medians to `benchmark/results/julia_phase6.json` (new artifact).

**Documentation**
- [ ] Update `README.md` performance table once both targets met.
- [ ] Add “Benchmark Capture” instructions describing how to refresh `benchmark/results/*.json`.

---

## Near-Term Timeline
| Day | Task                                   | Notes                                     |
|-----+----------------------------------------+-------------------------------------------|
| 0-1 | Implement CI workflow skeleton         | Smoke-test on fork before protecting main |
| 1-2 | Wire benchmark/stress jobs + artifacts | Ensure LD_LIBRARY_PATH exported for Julia |
| 2-3 | Enable required-status checks          | Coordinate with repo admins               |

Performance gap closure runs in parallel; once metrics meet targets and CI is green, Phase 7 can be closed and Phase 8 (deployment) can start.

---

## Risks & Mitigations
| Risk                              | Likelihood | Impact | Mitigation                                    |
|-----------------------------------+------------+--------+-----------------------------------------------|
| CI timeouts on GPU jobs           | Medium     | Medium | Split workflows, use matrix + caching         |
| Benchmark noise on shared runners | High       | Medium | Add retries + record median of N runs         |
| Outstanding perf regressions      | Medium     | High   | Keep profiler traces attached to CI artifacts |

---

**Next checkpoint:** Re-evaluate once CI pipeline lands and benchmarks hit targets (<1 week). Completed artifacts stay in `project_schedule/completed/`.
