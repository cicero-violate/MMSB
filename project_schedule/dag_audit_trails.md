# DAG Plan — Audit Trails

This DAG defines the execution order for introducing **audit trails**
without contaminating judgment or execution semantics.

Audit trails are strictly **observational**.
They MUST NOT influence control flow.

---

## Nodes

### A0 — Identify Audit Surface
- Identify auditable events:
  - Judgment issuance
  - Commit requests
  - Delta application
- Explicitly exclude:
  - Optimization decisions
  - Adaptive heuristics

---

### A1 — Canonical Audit Sources
- Declare authoritative audit sources:
  - TLog (`01_page/tlog.rs`, `tlog_serialization.rs`)
  - Replay + validation (`tlog_replay.rs`, `replay_validator.rs`)
  - Provenance tracker (`06_utility/provenance_tracker.rs`)
  - Telemetry / logging (read-only)
- Explicitly forbid:
  - Audit data influencing execution, scheduling, or judgment

---

### A2 — Audit Record Normalization
- Normalize existing audit data into:
  - Immutable event records
  - Hash-linked ordering
- Derive from existing logs; do not duplicate storage

---

### A3 — Append-Only Enforcement
- Enforce append-only semantics over:
  - TLog writes
  - Provenance emission
- Add assertions:
  - No mutation paths
  - No deletion paths

---

### A4 — Non-Interference Proof
- Prove that:
  - Audit modules are not imported by execution-critical paths
  - No audit data is read by propagation, scheduling, or judgment
- Establish one-way dependency:
  - Execution → Audit

---

### A5 — Replay & Validation Tooling
- Use existing replay engines to:
  - Reconstruct event timelines
  - Verify ordering and integrity
  - Detect gaps or corruption

### A6 — Audit Boundary Lock
- Freeze audit interfaces
- Require explicit review for new audit fields

---

## DAG Edges

```
A0 → A1 → A2 → A3 → A4 → A5 → A6
```

---

## Invariants

- Audit trails are observational only
- No audit state may be read by execution logic
- Loss of audit data must not corrupt execution

---

## Completion Criteria

- All auditable events are logged
- Logs are immutable and replayable
- Execution semantics are unchanged
