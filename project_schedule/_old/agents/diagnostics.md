# Diagnostics Agent Playbook
## Charter
- Own the diagnostic execution loop (tests, targeted reproductions).
- Capture and annotate logs (`pending_work/diagnostic_output.log`, `DIAGNOSTICS.md`).
- Confirm blockers before escalating to Rust/Julia agents.

## Tool Belt
- `julia --project=. test/runtests.jl 2>&1 | tee pending_work/diagnostic_output.log`
- `rg -n "=== mmsb" pending_work/diagnostic_output.log` for quick log anchors.
- Optional: `pending_work/bin/diag-run.sh` (create when automation agent delivers).

## Standard Runbooks
1. Sync `libmmsb_core.so` if Rust changed (`cargo build --release && cp ...`).
2. Execute the immediate to-do sequence:
   - `julia --project=. test/runtests.jl 2>&1 | tee pending_work/diagnostic_output.log`
   - Note final Rust log line from `mmsb_page_read` in `DIAGNOSTICS.md`.
   - File/update issue describing the failure mode and link it from `DAG_PRIORITY.md`.
3. Record final hypothesis + evidence in `DIAGNOSTICS.md`.
4. Update `TASK_LOG.md` + `DAG_PRIORITY.md` (status/notes).

## Reporting
- Append each run under “Current Findings” in `DIAGNOSTICS.md`.
- Summaries go to `TASK_LOG.md` with Task ID `P8.3` or relevant.

## Open Questions / TODOs
- Automate diagnostics run/summary script.
