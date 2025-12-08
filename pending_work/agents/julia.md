# Julia Agent Playbook
## Charter
- Edit Julia runtime, types, and test harnesses (`src/**/*.jl`, `test/`).
- Enforce state machine + FFI safety invariants.
- Coordinate with Diagnostics and QA when changes affect behavior.

## Tool Belt
- `julia --startup-file=no --project examples/quickstart.jl` (sanity check).
- `julia --project=. test/runtests.jl` (smoke tests, when not owned by Diagnostics).
- `rg -n "rust_" src` to find FFI touchpoints.
- Preferred editor/IDE for Julia code.

## Standard Runbooks
1. Review `TASK_LOG.md` for outstanding Julia work (T1–T8 etc.).
2. Implement changes; add concise comments for complex logic.
3. Run targeted Julia scripts/tests relevant to the edit.
4. Update docs if invariants change (e.g., `DIAGNOSTICS.md`, `AGENTS.md`).
5. Append entry to `TASK_LOG.md` summarizing edits/tests.

## Reporting
- Mention affected files + verification steps in each log entry.
- Coordinate with Docs Agent if new guidance/playbooks needed.

## Open Questions / TODOs
- Formalize GC stress/fuzz harness ownership once Rust segfault resolved.
