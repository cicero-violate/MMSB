# DAG Plan — Proposer / Challenger Roles

This DAG defines the **non-authoritative cognitive roles**
of Proposer and Challenger operating strictly under Judgment.

This DAG does not introduce new authority.
It formalizes safe interaction patterns only.

---

## Role Definitions (Non-Authoritative)

- **Judgment**: Chooses frames, authorizes action, or refuses action
- **Proposer**: Generates candidate frames or actions
- **Challenger**: Stress-tests a given frame or proposal

Only Judgment has authority.

---

## Nodes

### J0 — Judgment Context Declaration
- Judgment explicitly defines:
  - Scope
  - Intent
  - Constraints
- No generation occurs before this step.

---

### J1 — Proposer Invocation
- Proposer generates:
  - Candidate actions
  - Candidate frames (if explicitly requested)
- Proposals must:
  - Be non-binding
  - Contain no ranking or selection

---

### J2 — Challenger Invocation (Explicit)
- Challenger is invoked only if requested by Judgment.
- Challenger may:
  - Identify assumptions
  - Surface risks
  - Enumerate failure modes
  - Compare against known alternative frames

---

### J3 — Challenger Output Constraint
- Challenger output must:
  - Contain critique only
  - Contain no veto language
  - Contain no action directives

---

### J4 — Judgment Resolution
- Judgment alone may:
  - Accept a proposal
  - Revise a proposal
  - Reject a proposal
  - Reject the entire frame
  - Refuse action entirely

No other node may terminate or authorize execution.

---

## DAG Edges

```
J0 → J1
J1 → J4
J1 → J2 → J3 → J4
```

---

## Invariants

- Proposer has no authority
- Challenger has no veto power
- Challenger is opt-in only
- Judgment is singular and final

---

## Completion Criteria

This DAG is complete when:

- No proposal can execute without Judgment
- No challenger output can block execution
- Judgment can refuse action at any point

