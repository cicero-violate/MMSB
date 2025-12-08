═══════════════════════════════════════════════════════════════════
MMSB AGENT PROMPTS - COPY/PASTE READY
═══════════════════════════════════════════════════════════════════

Working Directory: /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB

┌─────────────────────────────────────────────────────────────────┐
│ DIAGNOSTICS AGENT                                               │
└─────────────────────────────────────────────────────────────────┘

ROLE: Diagnostics Agent
CWD: /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB

Step 1: fs.read pending_work/agents/diagnostics.md
Step 2: shell.exec julia --project=. test/runtests.jl 2>&1 | tee pending_work/diagnostic_output.log
Step 3: shell.exec rg -n "=== mmsb" pending_work/diagnostic_output.log  
Step 4: fs.write pending_work/DIAGNOSTICS.md (append findings)
Step 5: fs.write pending_work/TASK_LOG.md (append diagnostic summary)

BEGIN EXECUTION.

┌─────────────────────────────────────────────────────────────────┐
│ PLANNING AGENT                                                  │
└─────────────────────────────────────────────────────────────────┘

ROLE: Planning Agent
CWD: /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB

Step 1: fs.read pending_work/agents/planning.md
Step 2: fs.read pending_work/DAG_PRIORITY.md
Step 3: fs.read pending_work/PROJECTS_SCHEDULE.md
Step 4: Identify highest priority task needing schedule update
Step 5: fs.write pending_work/DAG_PRIORITY.md (update status/dependencies)
Step 6: fs.write pending_work/PROJECTS_SCHEDULE.md (sync milestones)
Step 7: fs.write pending_work/TASK_LOG.md (append planning decision)

BEGIN EXECUTION.

┌─────────────────────────────────────────────────────────────────┐
│ RUST CORE AGENT                                                 │
└─────────────────────────────────────────────────────────────────┘

ROLE: Rust Core Agent
CWD: /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB

Step 1: fs.read pending_work/agents/rust_core.md
Step 2: fs.read pending_work/DAG_PRIORITY.md (identify Rust tasks)
Step 3: shell.exec cargo build --release 2>&1 | tee pending_work/rust_build.log
Step 4: If build fails, extract errors and update DIAGNOSTICS.md
Step 5: If build succeeds, shell.exec cp target/release/libmmsb_core.so {julia_lib_path}
Step 6: fs.write pending_work/TASK_LOG.md (append build status)

BEGIN EXECUTION.

┌─────────────────────────────────────────────────────────────────┐
│ JULIA AGENT                                                     │
└─────────────────────────────────────────────────────────────────┘

ROLE: Julia Agent
CWD: /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB

Step 1: fs.read pending_work/agents/julia.md  
Step 2: fs.read pending_work/DAG_PRIORITY.md (identify Julia tasks)
Step 3: Implement feature from task specification
Step 4: fs.write {julia_source_file} (apply patch with new code)
Step 5: shell.exec julia --project=. -e 'using Pkg; Pkg.test()'
Step 6: fs.write pending_work/TASK_LOG.md (append implementation summary)

BEGIN EXECUTION.

═══════════════════════════════════════════════════════════════════
USAGE:
1. Copy the role-specific block above
2. Paste into ChatGPT Actions interface
3. The agent will immediately start executing from the specified CWD
═══════════════════════════════════════════════════════════════════

NOTES:
- CWD is explicitly set in each prompt
- All file paths are relative to CWD
- shell.exec commands run in CWD context
- OpenAPI schema provides fs.read, fs.write (with apply_patch), shell.exec tools
