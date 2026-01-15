# DAG Plan — Formal Verification Layer

This DAG defines the **selective formal proof layer**
protecting high-risk and irreversible boundaries.

Formal verification is **not global**.
It exists only where failure is unacceptable.

---

## Nodes

### F0 — Identify Proof Boundaries
- Enumerate proof-required transitions:
  - Judgment → Commit
  - Commit → Irreversible Delta
- Anchor to existing proof-adjacent modules:
  - `admission_proof.rs`
  - `execution_proof.rs`
- Explicitly exclude:
  - Adaptive rewrites
  - GPU scheduling
  - Performance heuristics

---

### F1 — Invariant Canonicalization
- Promote existing checks to named invariants:
  - Determinism (DAG acyclicity, semiring purity)
  - Authorization (judgment gate)
  - Integrity (page + delta integrity)
- Each invariant must define:
  - Scope
  - Failure mode
  - Enforcement boundary

---

### F2 — Minimal Formal Model
- Model only:
  - Judgment
  - Commit
  - Irreversible delta application
- Explicitly ignore:
  - GPU behavior
  - Memory layout
  - Adaptive optimization

---

### F3 — Proof System Selection
- Select proof tooling (e.g. Lean):
  - Capable of machine-checked proofs
  - External to runtime
- No runtime proof execution.

---

### F4 — Proof Lifting
- Lift existing runtime checks into:
  - Formal proof obligations, or
  - Explicit proof assumptions
- Runtime checks remain as defensive mirrors

---

### F5 — Proof Gate Integration
- Require proof artifacts for irreversible transitions only
- Gate behavior:
  - Missing proof → block irreversible action
  - Proof failure → explicit stop

### F6 — Proof Drift Detection
- Detect when code changes invalidate proofs
- Proof mismatches must fail loudly

---

## DAG Edges

```
F0 → F1 → F2 → F3 → F4 → F5 → F6
```

---

## Invariants

- Proofs guard transitions, not execution
- Judgment remains external to proofs
- Exploratory work remains unblocked

---

## Completion Criteria

- All irreversible paths are proof-gated
- Proof failures are explicit and non-silent
- Execution outside proof scope remains free
