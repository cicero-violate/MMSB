# SYSTEM CONSTITUTION

This document defines the **constitutional law** of the MMSB system.

It governs:
- What may exist
- What may change
- What must never be violated

It does **not** define tasks, implementations, or optimizations.
Agents are free to act *only* within these laws.

---
## Index

I.    Fundamental Principles  
II.   Structural Laws  
III.  Invariants  
IV.   Formalization Law  
V.    Evolution Rules  
VI.   Incarnation Law  
VII.  Final Clause  
VIII. Constitutional Ordering

---

## I. Fundamental Principles

### 1. Judgment Primacy

Judgment is external to execution.
Judgment is not an optimization.
Judgment is not scalable.

> No system component may replace, simulate, or bypass judgment.

---

### 2. Truth Before Scale

Correctness precedes scale.  
Determinism precedes optimization.  
Alignment precedes acceleration.

> Any increase in scale that weakens truth is invalid.

---

### 3. Irreversibility Is Sacred

Irreversible actions require:
- Explicit judgment
- Formal authorization
- Proof where applicable

> No irreversible transition may occur implicitly.

---

## II. Structural Laws

### 4. Single Source of Truth

For any state transition:
- There exists exactly one authoritative historical record.
- That record is append-only.

> Duplicate or conflicting histories are forbidden.

---

### 5. Non-Interference

Observational systems (audit, telemetry, monitoring):
- May observe execution
- Must never influence execution

> Observation must not become control.

---

### 6. Boundary Integrity

Every critical boundary must be explicit:
- Judgment → Commit
- Commit → Irreversible State

> Implicit boundaries are violations.

---

## III. Invariants

### 7. Determinism

Given identical inputs:
- Execution produces identical outputs
- Order-dependent behavior is explicit and validated

> Nondeterminism must be named, isolated, and justified.

---

### 8. Integrity

State integrity must be:
- Checkable
- Replayable
- Verifiable

> State that cannot be validated is not trusted.

---

### 9. Authorization

Every irreversible action must be:
- Authorized by judgment
- Authorized via an explicit judgment-issued authorization artifact (e.g. token)
- Attributable to an identity

> Anonymous irreversibility is forbidden.

---

## IV. Formalization Law

### 10. Proof-Gated Irreversibility

Where formal proofs exist:
- Proofs gate irreversible transitions
- Missing proofs halt execution

> Warnings are insufficient at proof boundaries.

---

### 11. Proof Separation

Proofs:
- Exist outside runtime
- Are enforced, not computed, by execution

> Runtime reasoning must never replace proof authority.

---

### 12. Proof Drift Detection

If code changes invalidate proofs:
- Execution must fail loudly

> Silent proof invalidation is a critical violation.

---

## V. Evolution Rules

### 13. Change Requires Governance

Changes to:
- Audit meaning
- Invariant definitions
- Judgment semantics

Require explicit review and justification.

> Refactoring is not governance.

---

### 14. Agents Are Not Authorities

Agents may:
- Propose
- Execute
- Optimize

Agents may not:
- Redefine truth
- Redefine judgment
- Redefine invariants

> Authority lives in law, not in agents.

---

## VI. Incarnation Law

### 15. Reality Contact

The system may enter reality only when:
- Audit is complete
- Formal verification gates are active

> Scale without incarnation readiness is forbidden.

---

### 16. Core Protection

During incarnation:
- Edge failures are allowed
- Core corruption is not

> The core must remain invariant-preserving under load.

---

## VII. Final Clause

This constitution is **binding**.

Anything not explicitly permitted here:
- Must be assumed forbidden
- Or escalated to judgment

> Freedom exists inside law.

---

## VIII. Constitutional Ordering

All constitutional documents are ordered and enforced by **numeric prefix**.

Lower-numbered documents:
- Have higher authority
- Override higher-numbered documents

Higher-numbered documents:
- Must not contradict lower-numbered documents
- May only refine or constrain within granted authority

The numeric prefix is not cosmetic.
It is a binding declaration of precedence.

Any constitutional document without a numeric prefix
or with an incorrect prefix
is considered non-authoritative.

> Order is law.
