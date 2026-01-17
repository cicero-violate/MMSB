## Canonical Structural vs State Pipelines (MMSB)

This document describes the **internal pipelines executed by mmsb-memory**
after approval by Judgment.

All routing and authorization decisions are made **before** these pipelines
via the ExecutionPlan emitted by mmsb-judgment.

## External Approval Flow (Shared)
```

Intent
↓
Policy
↓
Judgment
├─ emits JudgmentProof (C)
└─ emits ExecutionPlan (structural | state)

```

From this point onward, execution is **mechanical only**.
---

## STRUCTURAL PIPELINE (inside mmsb-memory)

```

StructuralIntent
↓
StructuralOps
↓
ShadowPageGraph (apply ops)
↓
Structural Validation

* acyclic
* reference integrity
* graph invariants
  ↓
  AdmissionProof (D)
  ↓
  commit_structural_delta
  ↓
  CommitProof (E)
  ↓
  DependencyGraph snapshot updated
  ↓
  OutcomeProof (F)
```

Notes:
- Structural validation is performed by memory, not judgment
- No routing or approval occurs here
- All commits require a verified JudgmentProof (C)

---

## STATE PIPELINE (inside mmsb-memory)
```

StateIntent
↓
Delta
↓
Delta Validation

* schema
* bounds
* epoch consistency
  ↓
  AdmissionProof (D)
  ↓
  commit_delta → tlog
  ↓
  CommitProof (E)
  ↓
  snapshot DependencyGraph
  ↓
  propagation
  ↓
  OutcomeProof (F)

```

Notes:
- Propagation happens only after commit
- Executor never commits directly
- All mutations are sealed by memory
---

## Proof Requirements (Mandatory)
Every execution MUST satisfy:
```
IntentProof (A)
⊂ PolicyProof (B)
⊂ JudgmentProof (C)
⊂ AdmissionProof (D)
⊂ CommitProof (E)
⊂ OutcomeProof (F)
```

Missing or invalid proofs invalidate the execution.

---

## Authority Invariants

- Judgment is the only approval authority
- Memory is the only commit authority
- Executors are mechanical
- Pipelines do not decide


