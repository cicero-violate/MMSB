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

- [ ] Raise multi-thread throughput ≥10 M/sec (profiling + batching plan).
- [ ] Bring Julia tick median <16 ms (profiled hotspots + logging knobs).
- [ ] Document tuning steps in `README.md` after targets met.

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
