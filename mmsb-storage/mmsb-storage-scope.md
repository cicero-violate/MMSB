# mmsb-storage

## Purpose
Provide durable, append-only persistence for MMSB artifacts.

This module stores bytes. It does not understand meaning.

---
## Definitions
- IntentProof (A)
- PolicyProof (B)
- JudgmentProof (C)
- AdmissionProof (D)
- CommitProof (E)
- OutcomeProof (F)
- KnowledgeProof (G)
- ProofChain definitions
- Proof graph semantics

## Responsibilities

- Persist all MMSB artifacts durably:
  - Proofs (A → G)
  - Events
  - Deltas
  - Snapshots
  - Knowledge records
- Provide ordered, append-only write semantics
- Provide read APIs for replay, audit, and recovery
- Ensure durability, integrity, and availability
- Support multiple backends (filesystem, object store, DB)

---

## Owns

- Physical layout of stored data
- Storage backend adapters
- Append-only logs
- Snapshot persistence
- Retention and compaction policies (mechanical only)

---

## Does NOT Do

- No interpretation of data
- No proof validation
- No authority decisions
- No judgment
- No policy logic
- No execution logic
- No learning logic
- No mutation of canonical truth

---

## Guarantees

- Written data is not modified in place
- Reads return exactly what was written
- Ordering is preserved per log
- Data is retrievable for full replay
- Storage failures do not change semantics

---

## Authority

- NONE

mmsb-storage is a durability substrate, not a decision-maker.
