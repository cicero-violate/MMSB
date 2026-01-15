# Audit Trails — Constitutional Map

This document defines the **legal boundaries, domain mappings, and exit criteria**
for the Audit Trails DAG.

It does not prescribe implementation.
It defines what must be true.

---

## DAG Node → Domain Mapping

### A0 — Identify Audit Surface
**Concern:** What is observable truth?

Indicative domains:
- `01_judgement/` — judgment issuance
- `01_page/tlog.rs` — irreversible state history
- `01_page/page_commit.rs` — commit boundary
- `04_propagation/tick_orchestrator.rs` — execution boundary (observe only)

**Boundary Law**
> Anything that changes state but cannot be replayed is not auditable.

---

### A1 — Canonical Audit Sources
**Concern:** What is authoritative?

Authoritative domains:
- `01_page/tlog*.rs`
- `06_utility/provenance_tracker.rs`
- `ReplayEngine.jl`, `replay_validator.rs`

Non-authoritative:
- `logging.rs`
- `telemetry.rs`

**Boundary Law**
> There must exist exactly one source of historical truth per state transition.

---

### A2 — Audit Record Normalization
**Concern:** One event, one meaning.

Indicative domains:
- `tlog_serialization.rs`
- `provenance_tracker.rs`
- `telemetry.rs` (read-only, lossy)

**Boundary Law**
> Normalization may derive, never invent.

---

### A3 — Append-Only Enforcement
**Concern:** Time only moves forward.

Indicative domains:
- `tlog.rs`
- `tlog_replay.rs`
- `ReplayEngine.jl`

**Boundary Law**
> Any mutation path is a violation, even if considered safe.

---

### A4 — Non-Interference Proof
**Concern:** Audit must never act.

Domains that must not depend on audit:
- `04_propagation/`
- `05_adaptive/`
- `02_semiring/`
- `03_dag/`

**Boundary Law**
> Audit observes execution; execution must be ignorant of audit.
> Audit must not influence judgment issuance, authorization, or policy evaluation.

---

### A5 — Replay & Validation
**Concern:** Truth can be reconstructed.

Indicative domains:
- `replay_validator.rs`
- `graph_validator.rs`
- `ShadowPageGraph.jl`

**Boundary Law**
> If history cannot be replayed, it was never truth.

---

### A6 — Audit Boundary Lock
**Concern:** Stability of meaning.

Conceptually frozen:
- Event schemas
- Hash semantics
- Audit field meanings

**Boundary Law**
> Changing audit meaning requires governance, not refactoring.

---

## Exit Criteria (Audit Trails)

Audit Trails are considered complete **iff** all are true:

- A single append-only historical truth exists
- Execution semantics do not reference audit state
- Loss of audit data does not alter behavior
- Replay produces identical state transitions
- Audit evolution requires explicit judgment

Failure of any condition means audit is incomplete.
