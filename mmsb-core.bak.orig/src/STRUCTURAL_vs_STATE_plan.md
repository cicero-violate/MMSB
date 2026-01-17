Understood.
Here is the **rewritten, authoritative phase spec**, with **explicit rules** and a **clear pending / TODO list**.
No politeness. No filler.

---

# MMSB PHASES — CANONICAL, ENFORCED

This is the **only valid ordering**.
Any deviation is a bug.

---

## **PHASE 0 — INGEST (NON-SEMANTIC)**

**Purpose**
Normalize external artifacts into candidate inputs.

**Allowed**

* Parse JSON
* Hash content
* Order messages
* Temporary shadow pages
* Utility-only graphs

**Forbidden**

* ❌ DAG mutation
* ❌ Page state mutation
* ❌ Judgment
* ❌ Propagation
* ❌ Persistence of truth

**Outputs**

* Candidate Page Deltas
* Candidate StructuralOps
* Evidence blobs

**Rule**

> Phase 0 may lie. Nothing here is truth.

**Status**

* ✅ Implemented
* ❌ Overused (some code should be deleted later)

---

## **PHASE 1 — STRUCTURE (AUTHORITATIVE LAW)**

**Purpose**
Define causality and dependency.

**Allowed**

* StructuralOp creation
* Shadow graph mutation
* Cycle detection
* Reference validation
* Structural judgment
* Authoritative DAG commit
* Snapshot persistence

**Forbidden**

* ❌ Page data mutation
* ❌ Propagation
* ❌ State deltas
* ❌ Implicit structure inference

**Outputs**

* Versioned DependencyGraph snapshot

**Rule**

> Structure defines what *may* happen, not what *did* happen.

**Status**

* ✅ Implemented (you just did this)

---

## **PHASE 2 — STATE ADMISSION (INTENT LOCKING)**

**Purpose**
Approve state change intent.

**Allowed**

* Page Delta creation
* Admission proof verification
* JudgmentToken validation
* Append-only TLog write

**Forbidden**

* ❌ DAG mutation
* ❌ Page materialization
* ❌ Propagation
* ❌ State reads

**Outputs**

* Persisted deltas (intent only)

**Rule**

> Approved deltas are promises, not effects.

**Status**

* ✅ Implemented
* ⚠️ Needs DAG snapshot association

---

## **PHASE 3 — STATE MATERIALIZATION (PURE FUNCTION)**

**Purpose**
Compute page state from deltas.

**Allowed**

* Delta replay
* Merge logic
* Mask application
* Deterministic reconstruction

**Forbidden**

* ❌ Writes
* ❌ Judgment
* ❌ DAG reads
* ❌ Propagation

**Outputs**

* Materialized page views

**Rule**

> Materialization must be repeatable and side-effect free.

**Status**

* ⚠️ Partially implemented
* ❌ Not cleanly isolated

---

## **PHASE 4 — PROPAGATION (DERIVED EFFECTS)**

**Purpose**
Compute downstream consequences.

**Allowed**

* Read-only DependencyGraph
* Descendant traversal
* Derived delta creation
* commit_delta (again)

**Forbidden**

* ❌ DAG mutation
* ❌ Structural ops
* ❌ Judgment skipping
* ❌ Inference of dependencies

**Outputs**

* Derived page deltas

**Rule**

> Propagation obeys structure; it never changes it.

**Status**

* ⚠️ Exists but NOT wired to authoritative DAG
* ❌ Currently ornamental

---

## **PHASE 5 — EXECUTION PROOF (EVIDENCE)**

**Purpose**
Prove that side effects occurred.

**Allowed**

* Read external outputs
* Hash results
* Attach execution proofs

**Forbidden**

* ❌ Mutation
* ❌ Propagation
* ❌ Structure changes

**Outputs**

* ExecutionProof streams

**Rule**

> Proofs confirm reality; they do not create it.

**Status**

* ✅ Implemented
* ⚠️ Overloaded with utility concerns

---

## **PHASE 6 — ACCELERATION (OPTIONAL)**

**Purpose**
Performance only.

**Allowed**

* GPU kernels
* NCCL
* Batching
* Caching

**Forbidden**

* ❌ Semantic change
* ❌ Skipping judgment
* ❌ Non-determinism

**Outputs**

* Faster execution

**Rule**

> Acceleration must be observationally invisible.

**Status**

* ❌ Not wired
* ❌ Safe to ignore for now

---

## **PHASE 7 — ADAPTATION (PROPOSALS ONLY)**

**Purpose**
Suggest better structure.

**Allowed**

* Analyze DAG
* Propose StructuralOps

**Forbidden**

* ❌ Auto-commit
* ❌ Silent changes

**Outputs**

* Candidate StructuralOps → Phase 1

**Rule**

> Adaptation proposes; judgment disposes.

**Status**

* ❌ Not started
* ❌ Must be human/LLM-assisted

---


