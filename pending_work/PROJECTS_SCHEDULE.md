# MMSB Project Schedule & Milestones
_Last updated: 2025-12-06 19:00 UTC_

## Release Target
- **Tag**: `v0.1.0-alpha` (current) with Rust-backed propagation.
- **Next Gate**: P8.4 (Rust segfault fix) → Beta-quality replay and checkpoint pipeline.
- **Stretch Goals**: GPU allocator (P9), CI hardening (P10), release automation.

## Near-Term Calendar (rolling 2 weeks)
| Day     | Focus                  | Owner             | Deliverable                                        |
| ---     | ---                    | ---               | ---                                                |
| Day 0   | Run diagnostics (P8.3) | Diagnostics Agent | `diagnostic_output.log` with Rust log trail        |
| Day 1   | Analyze logs           | Rust Agent        | Root-cause note + failing frame                    |
| Day 2-3 | Implement fix          | Rust Agent        | Updated `libmmsb_core.so`, regression plan         |
| Day 4   | Verification pass      | QA Agent          | `Pkg.test()` clean + GC stress + sanitizer dry run |
| Day 5   | Documentation sync     | Docs Agent        | Update `DIAGNOSTICS.md`, `AGENTS.md`, changelog    |
| Day 6+  | Contingency            | All               | Iterate if fix requires more cycles                |

> **Note**: Shift calendar forward if diagnostics uncover new blockers.

## Rolling Milestones
1. **M1 – Diagnostic Evidence (P8.3)**  
   - Artifact: `diagnostic_output.log`, annotated failure mode.  
   - Exit: Confident pointer to failing Rust statement.

2. **M2 – Rust Fix & Verification (P8.4)**  
   - Artifact: Passing Julia tests, clean sanitizer run, update to `RELEASE_NOTES.md`.  
   - Exit: No segfault, regression tests cover failure.

3. **M3 – Event/Test Hardening (T8–T10)**  
   - Artifact: Expanded test suites (`gc_stress_test.jl`, fuzzers), documented sanitizer baseline.  
   - Exit: Repeatable passes with stress harness.

4. **M4 – GPU & CI Tracks (P9–P10)**  
   - Artifact: GPU allocator implementation + CI pipeline gating on sanitizers.  
   - Exit: GPU path feature-flagged, CI prevents regression.

## Backlog Buckets
- **Diagnostics Backlog**: Additional instrumentation, targeted unit tests for `page.data_slice`, allocator probes.
- **Productization**: GPU memory support, propagation batching improvements, release automation scripts.
- **Quality**: CI gating, fuzzing, long duration stress, documentation of GC best practices.

## Risk Log
| Risk                                 | Impact        | Mitigation                                           |
| ---                                  | ---           | ---                                                  |
| Diagnostics inconclusive             | Delays fix    | Iterate logging granularity, capture core dumps      |
| Allocator corruption deeper in stack | Larger fix    | Bisect via allocator-only tests, add Rust unit tests |
| GPU work blocked by CPU instability  | Schedule slip | Keep GPU tasks behind flag until CPU path stable     |
| CI gaps allow regressions            | Quality risk  | Prioritize sanitizer gating after fix                |

## Hand-off Checklist Per Milestone
- Update `AGENTS.md` with new commands/expectations.
- Append summaries to `DIAGNOSTICS.md` when new evidence appears.
- Record timeline adjustments here to keep everyone aligned.

