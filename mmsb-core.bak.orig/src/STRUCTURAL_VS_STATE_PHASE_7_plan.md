Below is a **concrete, actionable recommendation for Phase 7** that fits the system you’ve built and **does not reopen any core phase**.

No theory. This is an execution plan.

---

# Phase 7 — Adaptive Structural Proposals (Recommended Design)

## Purpose (non-negotiable)

Phase 7 exists to **propose improvements to structure**, not to apply them.

> Phase 7 may *observe everything*
> Phase 7 may *change nothing*

All outputs must route back through **Phase 1 (structural judgment)**.

---

## Hard Rules (lock these first)

1. **Phase 7 produces proposals only**
2. **No mutation of DAG**
3. **No mutation of state**
4. **No propagation**
5. **No commits**
6. **No side effects**
7. **All outputs are discardable**

If any rule is violated, Phase 7 is invalid.

---

## What Phase 7 Consumes (Inputs)

Phase 7 is allowed to read:

### Authoritative Inputs

* `DependencyGraph` (read-only)
* Structural snapshots
* Page IDs
* Edge types

### Observational Inputs

* Propagation fanout counts
* Delta frequency per page
* Materialization cost stats
* Historical derived-delta counts

### Optional (later)

* Telemetry
* Timing data
* Memory locality stats

---

## What Phase 7 Produces (Outputs)

Phase 7 outputs **only** this:

```rust
struct StructuralProposal {
    proposal_id: ProposalID,
    ops: Vec<StructuralOp>,
    rationale: String,
    expected_effect: ExpectedEffect,
    confidence: f32,
}
```

Nothing else.

### Example `StructuralOp` outputs

* AddEdge
* RemoveEdge
* Re-type edge
* Split node
* Merge nodes

These are **candidates**, not actions.

---

## Core Proposal Types (start with these 4 only)

Do NOT overbuild. Start here.

### 1. High-Fanout Reduction

Detect pages where:

* fanout >> median
* delta frequency is high

Propose:

* splitting dependencies
* introducing intermediary nodes

---

### 2. Dead Dependency Elimination

Detect edges where:

* no downstream deltas observed over window N

Propose:

* RemoveEdge

---

### 3. Locality Optimization

Detect:

* propagation chains with poor locality
* cross-module dependencies

Propose:

* restructure DAG to cluster locality

---

### 4. Structural Simplification

Detect:

* diamond dependency patterns
* redundant paths

Propose:

* edge removal or re-routing

---

## Phase 7 Execution Model

Phase 7 runs as a **batch analyzer**, not a daemon.

```
Snapshot DAG
   ↓
Collect stats
   ↓
Generate proposals
   ↓
Emit report
```

No feedback loop. No automation.

---

## Integration Point (exact)

Add **one new module**:

```
07_adaptive/proposal_engine.rs
```

Public API:

```rust
pub fn generate_structural_proposals(
    dag: &DependencyGraph,
    stats: &PropagationStats,
) -> Vec<StructuralProposal>
```

That’s it.

---

## How LLMs Fit (safely)

LLMs may be used **only** to:

* generate rationales
* rank proposal confidence
* suggest alternative ops

LLMs may **not**:

* apply proposals
* infer hidden structure
* bypass judgment
* mutate data

They are advisors, not actors.

---

## How Proposals Flow Back

```
Phase 7 Proposal
   ↓
Human / LLM Review
   ↓
Structural Judgment (Phase 1)
   ↓
commit_structural_delta (if approved)
```

Phase 7 never skips Phase 1.

---

## Success Criteria for Phase 7 (objective)

Phase 7 is successful if:

* Structural churn decreases
* Propagation fanout decreases
* Derived deltas per root delta decreases
* No regression in correctness
* No phase boundary violations

---

## What NOT to Do in Phase 7

Do **not**:

* auto-commit
* run continuously
* mix with GPU
* mix with propagation
* add heuristics to Phase 4
* let LLMs “edit code”

---

## Final Recommendation (clear)

**Build Phase 7 as a proposal generator only.**
Start with **high-fanout reduction** and **dead dependency detection**.
Keep it batch, read-only, and discardable.

This gives you:

* leverage
* intelligence
* safety
* and a reason to optimize later

If you want next, I can:

* write the exact proposal data structures
* design a scoring function
* or produce the Claude prompt for proposal generation
