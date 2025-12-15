# Week 26 – CLAUDE.md Compliance Report

All seven non-negotiable rules that apply to Phase 5 work have been verified via code inspection and the Week 24/25 integration tests (`test/test_week24_25_integration.jl`). No deviations were found and replay metadata remains deterministic.

## Verification Table

| Rule | Requirement                                              | Status | Evidence                                                                                                                                                                                                                                                      |
|  --- | ---                                                      | ---    | ---                                                                                                                                                                                                                                                           |
|    1 | Rust executes and enforces; no reasoning runs in Rust.   | ✅     | `DeltaRouter.route_delta!` calls `FFIWrapper.rust_delta_apply!` for execution (`src/02_semiring/DeltaRouter.jl:24-41`) while reasoning-only code stays in Julia (`src/07_intention/intent_lowering.jl`).                                                      |
|    2 | Julia reasons and plans; it never mutates state.         | ✅     | `IntentLowering.execute_upsert_plan!` only reads via `read_page`, evaluates predicates, and hands control back to `DeltaRouter` for execution (`src/07_intention/intent_lowering.jl:29-48`).                                                                  |
|    3 | All state changes travel through canonical deltas in L1. | ✅     | `API.update_page` diffs pages, builds a mask, and calls `DeltaRouter.create_delta`/`route_delta!` (`src/API.jl:84-117`); `delta_validation.rs` rejects malformed deltas before apply (`src/01_page/delta_validation.rs:1-14`).                                |
|    4 | Intent is distinct from execution and is persisted.      | ✅     | Upsert plans collect snapshots + metadata, only enqueue deltas after predicates pass, and persist the query/id in TLog (`src/07_intention/intent_lowering.jl:33-47`, `test/test_week24_25_integration.jl:45-109`).                                            |
|    5 | Deterministic replay.                                    | ✅     | Replay tests reconstruct final state, filter by metadata, and validate checkpoints with identical intent IDs (`test/test_week24_25_integration.jl:119-185`).                                                                                                  |
|    6 | No learning/backprop inside L0–L5.                       | ✅     | `rg -n "Flux" src/0[0-5]*` and `rg -n "backprop" src/0[0-5]*` return no hits. Core layers only expose allocators, deltas, DAGs, propagation, and adaptive layout heuristics (see `src/05_adaptive/EntropyReduction.jl` for analytic, non-learning gradients). |
|    7 | No cognition below L6.                                   | ✅     | Reasoning/planning modules start at `src/08_reasoning` and `src/09_planning`; lower layers (e.g., `src/06_utility`, `src/07_intention/UpsertPlan.jl`) handle scoring and plan definition without inference loops.                                             |

## Additional Notes

1. **Tests cover QMU separation.** The Week 24 INT.G1 testset prevents accidental Query/Mutate/Upsert blending by asserting that blocked plans leave state untouched and metadata persists (`test/test_week24_25_integration.jl:64-99`).
2. **Replay remains deterministic after checkpoints.** The Week 25 INT.G2 suite replays intents both with predicate filters and from checkpoints, proving metadata survives persistence (`test/test_week24_25_integration.jl:131-185`).
3. **Documentation synchronized.** `docs/QMU_API.md` now captures the Intent → Delta pipeline, reinforcing the architectural guarantees described above.
