# Future Works

This document captures **deliberately deferred upgrades** that are acknowledged, scoped, and intentionally not yet executed.
These items are pending by design to preserve correctness, judgment, and alignment before scale.

---

## Judgment & Decision Layer

### Selective Delegation (Model C)
- Introduce a formal delegation boundary where judgment may delegate execution to sub-agents under explicit constraints.
- Preserve a single human-anchored judgment root.
- Ensure delegation never bypasses the judgment gate.

### Judgment Classification
- Add explicit classification tags to judgments (e.g. `Irreversible`, `Reversible`, `Exploratory`).
- Enable downstream systems to reason about risk without re-evaluating intent.

### Proposer / Challenger Role Formalization
- Introduce non-authoritative cognitive roles:
  - Proposer (generation)
  - Challenger (stress-testing)
- Roles operate strictly under Judgment.
- No role may authorize, veto, or execute.

See: `dag_judgment_roles.md`

---

## Audit & Traceability

### Audit Trails (Outside Judgment)
- Implement append-only audit logs for:
  - Judgment issuance
  - Commit requests
  - Delta application
- Ensure audits are observational only and cannot influence judgment.

---

## Formal Verification Layer

### Selective Proof Layer Around the Gate
- Introduce a **formal proof boundary** (e.g. Lean-backed) around:
  - Judgment → Commit transitions
  - Invariant preservation
  - Irreversible execution paths
- Proofs should be:
  - Selective (not global)
  - Required only at high-risk boundaries
  - Non-blocking for exploratory work

---

## Execution Principles

- Correctness precedes scale
- Determinism precedes optimization
- Alignment precedes acceleration
- Judgment is external to optimization

---

## Status

- These items are **pending by choice**, not omission.
- Execution is gated until correctness pressure demands them.
- Revisit when:
  - Audit pressure increases
  - External users are introduced
  - Irreversible surface area expands

---

## Post-Formalization Phase (What Comes Next)

Once formalization is complete, the system is considered **safe to compound**.
The objective shifts from internal correctness to **external utility under load**.

### Execution Expansion
- Increase execution throughput only on proven-safe paths.
- Apply the system to real, non-self-referential tasks.
- Use execution pressure to surface invariant weaknesses.

### Selective Scale
- Scale execution, not judgment.
- Keep the judgment root singular and human-anchored.
- Allow failures at the edges; protect the core.

### External Utility
- Attach MMSB to an external domain:
  - another codebase
  - another user
  - another problem class
- Let reality, not reasoning, become the primary validator.

### Economic Contact
- Establish a real feedback loop:
  - intelligence → utility → value
- Even minimal real-world coupling outweighs further internal refinement.

### Explicit Non-Goals
- No new abstraction layers
- No meta-architecture work
- No agent proliferation

These were prerequisites. Not destinations.

---

## Phase Transition Summary

- **Before formalization**: “Can this system be trusted?”
- **After formalization**: “Can this system carry weight?”

The post-formalization phase is about **incarnation**:
execution in the world, under consequences, without losing alignment.
