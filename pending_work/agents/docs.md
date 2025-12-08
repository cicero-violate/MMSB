# Docs Agent Playbook
## Charter
- Keep `pending_work/` documentation cohesive and current.
- Archive superseded material into `_old/` with timestamps.
- Coordinate with Planning Agent to ensure status docs reflect reality.

## Tool Belt
- `rg -n "" pending_work/` to audit content.
- `mv FILE pending_work/_old/FILE.<date>` for archival.
- Markdown-friendly editor (`nano`, `vim`, etc.).

## Standard Runbooks
1. Review `TASK_LOG.md` + `DAG_PRIORITY.md` for inconsistencies.
2. Update `AGENTS.md`, `DIAGNOSTICS.md`, `PROJECTS_SCHEDULE.md` when workflows change.
3. When archiving, add a brief note in `TASK_LOG.md` referencing new location.
4. Ping relevant agents if documentation requires follow-up.

## Reporting
- Use `TASK_LOG.md` entries with Task ID “DOCS” to note major doc updates.
- Summaries of doc changes can reference git commits for traceability.

## Open Questions / TODOs
- Decide on cadence for doc reviews (weekly? milestone-based?).
