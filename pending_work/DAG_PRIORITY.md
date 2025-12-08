# MMSB Execution Graph & Priority Board
_Last updated: 2025-12-06 19:00 UTC_

## Current Context
- **Product**: MMSB (Memory-Mapped State Bus) v0.1.0-alpha — deterministic shared-memory fabric for CPU/GPU propagation.
- **Blocking issue**: Segfault inside `mmsb_page_read` (Rust) when Julia runs `test/runtests.jl`.
- **Recent progress**: Julia-side safety work (Tasks T1–T6) and Rust diagnostic instrumentation (Tasks T0.1–T0.4) are complete; crash persists in Rust.
- **Goal of this board**: Provide a single view of dependencies, priorities, and completion state for all active tasks in `pending_work/`.

## Critical Path Snapshot
| Phase                                   | Purpose                                                    | Status          | Blocking Dependency                            |
| ---                                     | ---                                                        | ---             | ---                                            |
| **P8.2 – Error Mapping**                | Julia↔Rust error plumbing                                  | ✅ Complete     | —                                              |
| **P8.3 – Test Execution & Diagnostics** | Run Julia suite with Rust logging to capture segfault site | 🔴 Blocked      | Requires reliable module load & diagnostic run |
| **P8.4 – Fix Failures**                 | Implement Rust-side fix once failure point known           | ⏳ Pending P8.3 | Needs diagnostic evidence                      |
| **P9 – GPU Implementation**             | Real GPU allocator + propagation                           | ⏸ Deferred      | Needs stable CPU/Rust pipeline                 |
| **P10 – CI Hardening**                  | Sanitizers, fuzzing, Miri/Valgrind gating                  | ⏸ Deferred      | Depends on P8.4                                |

## Dependency Graph (textual)
1. **Rust Diagnostics (T0.1–T0.4)** → unlocks trustworthy crash signals. ✅ Done.
2. **Julia FFI Safety (T1–T6)** → ensures crash is not GC/state related. ✅ Done.
3. **Event + Test Hardening (T8–T10)** → depends on root cause fix. ⏳.
4. **GPU/CI/Roadmap tasks** → depend on successful diagnostics + fix.

```
T0.x (Rust logging) ─┐
T1–T6 (Julia safety) ├─> Diagnostic Test Run (P8.3) ──> Root Cause Fix (P8.4) ──> Expanded Testing (T8–T10) ──> GPU/CI Tracks
                     └───────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Priority Buckets
| Priority           | Owner             | Tasks                                                                                                      | Status                          |
| ---                | ---               | ---                                                                                                        | ---                             |
| **Immediate (P0)** | Diagnostics Agent | - Run Julia test suite with logging<br>- Capture `diagnostic_output.log`<br>- Identify final Rust log line | 🔴 Blocked by next steps       |
| **Next (P1)**      | Rust Agent        | - Patch failing Rust path once evidence captured<br>- Re-run tests + update docs                           | ⏳ Pending P0                   |
| **Stabilize (P2)** | QA/Runtime Agent  | - T8 Event handler audit<br>- T9 Edge case tests<br>- T10 Memory sanitizer runs                            | 🟡 Blocked by fix              |
| **Stretch (P3)**   | Platform Agent    | - GPU allocator implementation<br>- CI hardening/fuzzing<br>- Release prep                                 | 🟣 Deferred                     |
| **Immediate (P0)** | Diagnostics Agent | - Run Julia test suite with logging<br>- Capture `diagnostic_output.log`<br>- Identify final Rust log line | 🔴 OPEN                         |
| **Next (P1)**      | Rust Agent        | - Patch failing Rust path once evidence captured<br>- Re-run tests + update docs                           | ⏳ Pending P0                   |
| **Stabilize (P2)** | QA/Runtime Agent  | - T8 Event handler audit<br>- T9 Edge case tests<br>- T10 Memory sanitizer runs                            | 🟡 Not started (blocked by fix) |
| **Stretch (P3)**   | Platform Agent    | - GPU allocator implementation<br>- CI hardening/fuzzing<br>- Release prep                                 | 🟣 Deferred                     |

## Task Ledger (rolled up)
- **Completed (✅)**: T0.1–T0.4, T1–T6.
- **In progress (🟡)**: T8 (Event audit) — waiting on crash fix context.
- **Not started (⚪)**: T9–T10 (tests + sanitizers), GPU/CI tracks.

### Task Log
- The full running ledger now lives in `pending_work/TASK_LOG.md`; append new updates there while keeping this board focused on priorities.

## Success Criteria to Exit P8.3
1. `diagnostic_output.log` captured with latest Rust build.
2. Final log line before crash mapped to one of the documented failure modes (see `DIAGNOSTICS.md`).
3. Issue ticket filed/updated with precise failing instruction.

## References
- `AGENTS.md` — onboarding + execution order.
- `PROJECTS_SCHEDULE.md` — calendar view and deliverable cadence.
- `DIAGNOSTICS.md` — commands + interpretation guide.
