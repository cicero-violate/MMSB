# Automation Agent Playbook
## Charter
- Build and maintain scripts/CI jobs that standardize repetitive workflows (diagnostics, status updates, testing).
- Keep tooling discoverable and documented.
- Partner with other agents to identify pain points.

## Tool Belt
- `pending_work/bin/` (create scripts like `diag-run.sh`, `task-add.sh`).
- Bash/Python/Justfile depending on needs.
- Access to CI configs (if applicable) in `.github/` or `ci/`.

## Standard Runbooks
1. Collect requirements from other agents (what should be automated?).
2. Prototype script under `pending_work/bin/`; include `--help`.
3. Document usage in `AGENTS.md` and relevant playbooks.
4. Version control scripts, and note updates in `TASK_LOG.md` with Task ID `AUTO`.

## Reporting
- Each new/updated automation tool gets a `TASK_LOG.md` entry (include command synopsis).
- Coordinate with Planning Agent if automation changes process expectations.

## Open Questions / TODOs
- Implement scripted diagnostic run + log summary.
- Explore simple lint/check script to verify docs (AGENTS, DAG, TASK_LOG) stay consistent.
