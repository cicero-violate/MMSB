# MMSB Project Documentation Index

## Quick Navigation

| Document                | Lines | Purpose                     | Read When                             |
|-------------------------+-------+-----------------------------+---------------------------------------|
| **README.md**           |    53 | Quick overview              | First time viewing                    |
| **_SUMMARY.md**         |   174 | Executive summary with math | Need high-level picture               |
| **ARCHITECTURE.md**     |   513 | Complete specification      | Planning work or understanding system |
| **PROJECT_SCHEDULE.md** |   188 | Timeline and phases         | Checking progress or planning sprints |
| **DAG_DEPENDENCIES.md** |   323 | Task dependencies           | Finding next task to work on          |
| **TASK_LOG.md**         |   371 | Detailed task tracking      | Updating or checking task status      |
| **AGENTS.md**           |   507 | AI agent workflow           | You're an AI agent working on MMSB    |

**Total:** 2,129 lines of structured planning

---

## Document Relationships

```
README.md (entry point)
    ↓
_SUMMARY.md (mathematical summary)
    ↓
ARCHITECTURE.md (detailed design)
    ├→ PROJECT_SCHEDULE.md (when to do it)
    └→ DAG_DEPENDENCIES.md (what order to do it)
           ↓
       TASK_LOG.md (tracking what's done)
           ↑
       AGENTS.md (how agents work)
```

---

## For Different Roles

### Project Manager
1. Read: README.md → PROJECT_SCHEDULE.md
2. Track: TASK_LOG.md weekly
3. Monitor: Phase exit criteria in PROJECT_SCHEDULE.md

### Software Architect
1. Read: ARCHITECTURE.md completely
2. Reference: Layer definitions and file organization tables
3. Validate: Cross-layer dependencies

### Human Developer
1. Read: ARCHITECTURE.md → DAG_DEPENDENCIES.md
2. Pick: Unblocked P0 task from checklist
3. Update: TASK_LOG.md when starting/completing

### AI Agent
1. Read: AGENTS.md (complete workflow)
2. Execute: Tasks from DAG_DEPENDENCIES.md
3. Update: TASK_LOG.md after every task
4. Reference: ARCHITECTURE.md for context

### Technical Lead
1. Read: All documents
2. Focus: Critical path in DAG_DEPENDENCIES.md
3. Review: Weekly progress in TASK_LOG.md
4. Adjust: PROJECT_SCHEDULE.md if needed

---

## Key Numbers

- **Layers:** 13 (0-12)
- **Phases:** 4 (20 weeks)
- **Tasks:** 180+
- **Files:** 116 target (59 current)
- **Languages:** Rust (41), Julia (74), CUDA (1)
- **Docs:** 71KB total planning

---

## Common Queries

**"What layer am I working on?"**
→ Check PROJECT_SCHEDULE.md current week

**"What task should I do next?"**
→ Find unblocked P0 in DAG_DEPENDENCIES.md

**"Where does this file go?"**
→ Look up in ARCHITECTURE.md tables

**"What's the current status?"**
→ Check TASK_LOG.md recent entries

**"How do I work as an agent?"**
→ Read AGENTS.md workflow section

**"Why this architecture?"**
→ See ARCHITECTURE.md mathematical foundations

**"When are we done?"**
→ PROJECT_SCHEDULE.md Phase 4 exit criteria

---

## Document Updates

All documents are version-controlled and should be updated as:
- **ARCHITECTURE.md:** Only for major design changes
- **PROJECT_SCHEDULE.md:** Weekly if timeline shifts
- **DAG_DEPENDENCIES.md:** When dependencies change
- **TASK_LOG.md:** After every task start/completion
- **AGENTS.md:** When workflow patterns change

**Current Version:** v2.0 (13-layer architecture with reasoning/planning)
**Last Updated:** 2025-12-09

