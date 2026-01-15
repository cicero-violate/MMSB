# Formal Verification — Constitutional Map

This document defines the **legal boundaries, invariant domains, and exit criteria**
for the Formal Verification DAG.

It does not prescribe proofs or tooling.
It defines what must never be violated.

---

## DAG Node → Domain Mapping

### F0 — Identify Proof Boundaries
**Concern:** Where failure is unacceptable.

Indicative domains:
- `mmsb-judgment` (judgment issuance boundary)
- `01_page/page_commit.rs`
- `06_utility/admission_proof.rs`
- `06_utility/execution_proof.rs`

**Boundary Law**
> Only irreversible transitions deserve proof.

---

### F1 — Invariant Canonicalization
**Concern:** Name what must never break.

Invariant domains:
- Determinism:
  - `03_dag/graph_validator.rs`
  - `02_semiring/purity_validator.rs`
- Integrity:
  - `01_page/integrity_checker.rs`
- Authorization:
  - `mmsb-judgment` (judgment issuance boundary)

**Boundary Law**
> An unnamed invariant is not an invariant.

---

### F2 — Minimal Formal Model
**Concern:** Truth, not implementation.

Explicitly excluded:
- `00_physical/`
- `04_propagation/`
- `05_adaptive/`

**Boundary Law**
> If a detail is not required to state correctness, it must be ignored.

---

### F3 — Proof System Separation
**Concern:** External authority.

Conceptual scope:
- Proofs live outside runtime
- Proofs exist as artifacts

**Boundary Law**
> Runtime may enforce proofs, but never compute them.

---

### F4 — Proof Lifting
**Concern:** Turn checks into guarantees.

Indicative domains:
- `admission_proof.rs`
- `execution_proof.rs`
- `invariant_checker.rs`

**Boundary Law**
> Runtime checks are defensive shadows of formal truth.

---

### F5 — Proof Gate Integration
**Concern:** Hard stops.

Gate boundaries:
- Judgment → Commit
- Commit → Irreversible delta

**Boundary Law**
> Missing proof must stop reality, not warn it.

---

### F6 — Proof Drift Detection
**Concern:** Truth must stay true.

Impacted domains:
- Proof artifacts
- Boundary code
- Invariant definitions

**Boundary Law**
> A proof that no longer applies must fail loudly.

---

## Exit Criteria (Formal Verification)

Formal verification is complete **iff** all are true:

- Every irreversible transition is proof-gated
- Exploratory paths remain unblocked
- Proof failures halt execution deterministically
- Runtime checks mirror, not replace, proofs
- Code changes cannot silently invalidate proofs

Failure of any condition means formalization is incomplete.
