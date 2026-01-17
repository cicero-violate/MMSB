# mmsb-authenticate

## Purpose
Verify that MMSB proofs are valid according to the canonical definitions.

This module answers: “Does this proof satisfy the required rules?”

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

- Verify proof hash integrity
- Verify proof chain continuity (A ⊂ B ⊂ C ⊂ …)
- Verify signatures and authority markers (e.g. JudgmentProof)
- Verify epoch, replay, and admission constraints
- Verify invariant attestations produced by memory
- Reject malformed, incomplete, or forged proofs

---

## Owns

- Proof verification logic
- Cryptographic validation
- Chain consistency checks
- Replay and epoch guards
- Invariant verification hooks

---

## Does NOT Do

- No proof structure definitions
- No proof creation
- No authority or approval decisions
- No execution
- No memory mutation
- No learning logic

---

## Guarantees

- Proofs are either valid or rejected
- No implicit trust between modules
- Verification is deterministic and reproducible

---

## Authority

- NONE

mmsb-authenticate enforces correctness, not permission.
