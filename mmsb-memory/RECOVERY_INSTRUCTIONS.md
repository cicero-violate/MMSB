# MMSB Memory Recovery Instructions

## What Happened
Previous refactor used bash scripts with `sed`/`awk` that deleted closing braces throughout codebase.
All source files were corrupted. Directory structure recreated but files are blank.

## Current State
```
mmsb-memory/src/
├── admission/
├── truth/
├── delta/
├── structural/
├── dag/
├── commit/
├── page/
├── epoch/
├── tlog/
├── replay/
├── outcome/
├── propagation/
├── materialization/
├── semiring/
├── physical/
├── optimization/
├── device/
├── proofs/
├── lib.rs (blank)
├── types.rs (blank)
└── memory_engine.rs (blank)
```

## Recovery Task

**Copy clean source files from `mmsb-core` to `mmsb-memory`.**

The original working code is in:
```
/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/mmsb-core/
```

## Instructions for Next Claude

1. List all `.rs` files in `mmsb-core/src/`
2. Copy each file to corresponding location in `mmsb-memory/src/`
3. Verify files are intact (check for closing braces)
4. Run `cargo check` to confirm compilation
5. Git commit: "restore: recover mmsb-memory from mmsb-core"

## Key Files to Copy
- lib.rs
- types.rs  
- memory_engine.rs
- All subdirectories with their mod.rs and implementation files

## DO NOT
- Use sed, awk, or batch text processing scripts
- Modify file contents during copy
- Run any bash scripts that touch multiple files

## After Recovery
Resume Phase 6 refactor following TODO-dag.md, using ONLY apply_patch for individual file edits.
