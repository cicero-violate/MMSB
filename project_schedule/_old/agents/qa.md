# QA Agent Playbook
## Charter
- Validate the end-to-end system via `Pkg.test()`, stress suites, fuzzers, and sanitizers.
- Capture failures with sufficient logs to reproduce.
- Signal readiness for milestone exits (P8.4+, T9–T10).

## Tool Belt
- `julia --startup-file=no --project -e 'using Pkg; Pkg.test()'`
- `julia --project=. test/gc_stress_test.jl` (once committed)
- `julia --project=. test/fuzz_replay.jl` / other fuzz runners
- `valgrind --leak-check=full julia --project=. test/runtests.jl` (after fix)

## Standard Runbooks
1. Confirm `libmmsb_core.so` + Julia sources are current.
2. Run baseline tests; store logs under `pending_work/test_output.log`.
3. Execute stress/fuzz suites serially; note environment/config.
4. If failures occur, capture stack traces + annotate in `DIAGNOSTICS.md`.
5. Update `TASK_LOG.md` with pass/fail + next actions.

## Reporting
- Maintain checklist for T9/T10 completion.
- Communicate blockers to Planning Agent ASAP.

## Open Questions / TODOs
- Finalize sanitizer matrix (ASan, ThreadSan, Valgrind).
- Determine cadence for long-haul fuzz runs.
