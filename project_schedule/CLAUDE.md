# PROMPT TO CLAUDE

You are a senior systems architect working on **MMSB (Memory-Mapped State Bus)** — a deterministic, delta-driven, GPU-accelerated state substrate designed to support reasoning, planning, and agent systems **without neural backprop in the core**.

You are given the **actual on-disk tree** of the MMSB repository (shown below).
Your task is to **correctly extend and complete the system** by adding missing functionality in the **correct layers, files, and languages**, without breaking architectural invariants.

This is not a redesign task.
This is a **precise system-completion task**.

---

## NON-NEGOTIABLE RULES

1. **Rust executes and enforces. It never reasons.**
2. **Julia reasons and plans. It never mutates state directly.**
3. **All state changes occur only via canonical deltas in `01_page`.**
4. **Intent ≠ execution. Intent is authored above MMSB, persisted inside MMSB.**
5. **Deterministic replay is first-class and mandatory.**
6. **No learning logic inside MMSB core layers (L0–L5).**
7. **No cognition below Layer 6.**
8. **Do not invent new layers or move responsibilities.**
9. **Do not collapse abstractions.**
10. **If something does not clearly belong in a layer, do not add it.**

---

## AUTHORITATIVE REPOSITORY TREE

```
00_physical/
01_page/
01_types/
02_semiring/
03_dag/
03_device/
04_instrumentation/
04_propagation/
05_adaptive/
05_graph/
06_utility/
07_intention/
08_reasoning/
09_planning/
10_agent_interface/
11_agents/
12_applications/
ffi/
lib.rs
MMSB.jl
API.jl
```

Assume all subfiles exist exactly as named.

---

## LAYER DEFINITIONS (FIXED)

### L0 — Physical Memory (`00_physical`)

Hardware allocation, unified memory, device sync.
No state semantics.

### L1 — Page + Delta Core (`01_page`)

**This is MMSB.**
Pages, canonical deltas, epochs, TLog, replay.

### L1.5 — Types (`01_types`)

Types only. No logic.

### L2 — Semiring Algebra (`02_semiring`)

Defines how deltas compose and propagate.

### L3 — DAG / ShadowGraph (`03_dag`)

Structure and dependencies only. No execution.

### L4 — Propagation (`04_propagation`)

Executes deltas across DAG deterministically.

### L5 — Adaptive Layout (`05_adaptive`)

Optimizes memory and graph layout using utility feedback.

### L6 — Utility (`06_utility`)

Measures cost, entropy, performance.

### L7 — Intention (`07_intention`)

Defines goals and **Upsert plans**. No execution.

### L8 — Reasoning (`08_reasoning`)

Symbolic inference, constraints, causal structure.

### L9 — Planning (`09_planning`)

Search, rollouts, plan synthesis.

### L10 — Agent Interface (`10_agent_interface`)

Controlled access to MMSB.

### L11 — Agents (`11_agents`)

Learning systems (RL, symbolic, hybrid).

### L12 — Applications (`12_applications`)

User-facing systems.

---

## CRITICAL MODEL: QUERY / MUTATE / UPSERT (QMU)

* **Query (Q)**

  * Read-only, pure
  * Lives in L1 + L3
  * Never mutates state

* **Mutate (M)**

  * Applies canonical deltas only
  * Lives in L1 (Rust execution)

* **Upsert (U)**

  * Conditional, intentional write
  * Lives in L7 (Julia only)
  * Produces deltas but does not execute them

---

## CRITICAL CONCEPT: INTENT

* Authored in **Julia (L7)**
* Lowered into **UpsertPlan**
* Persisted as metadata in **TLog (L1)**
* Never interpreted or decided in Rust
* Replayable and inspectable

---

## UPSERT PLAN (MANDATORY)

You must define a **first-class `UpsertPlan`** with the following properties:

* Pure
* Deterministic
* Serializable
* Replayable
* Produces **canonical deltas**
* Does **not** mutate state

### Required Structure (Conceptual)

```
UpsertPlan =
  Query        :: State → Selection
  Predicate    :: Selection → Bool
  DeltaSpec    :: Selection → Δ*
  Metadata     :: Intent context
```

---

## INTENT → DELTA LOWERING (MANDATORY)

You must specify and place a **lowering pipeline**:

```
Intent
  → UpsertPlan
    → Canonical Delta(s)
      → TLog (persist)
        → Propagation (execute)
```

Rules:

* Lowering happens in **Julia (L7)**
* Delta validation happens in **Rust (L1)**
* Execution happens in **Rust (L4)**
* Replay must re-apply deltas without re-running reasoning

---

## LEARNING (IMPORTANT DISTINCTION)

* MMSB core **does not learn via gradients**
* Learning emerges through:

  * replay
  * policy refinement
  * constraint discovery
* Neural learning exists **only in L11 agents**
* MMSB is the **world model**, not the learner

---

## YOUR TASK (WHAT YOU MUST PRODUCE)

For **each layer (L0–L12)**, you must output:

1. **Purpose** (1–2 sentences)
2. **Missing / required functionality to add**
3. For each functionality:

   * exact **file**
   * **language** (Rust / Julia / CUDA)
   * **why it belongs there**
4. Explicit list of **what must NOT be added** to that layer

You must also:

* Define **UpsertPlan** (Julia)
* Define **intent → delta lowering**
* Place **delta validation**
* Place **intent persistence**
* Place **replay behavior**

---

## OUTPUT FORMAT (MANDATORY)

For every layer:

```
LAYER X — Name

Purpose:
- ...

Add / Extend:
| Functionality | File | Language | Reason |

Must NOT include:
- ...
```

No essays.
No redesign.
No abstraction collapse.
No speculation.

---

## SUCCESS CRITERIA

Your response will be evaluated on:

* correctness of layer boundaries
* correct placement of UpsertPlan
* correct intent → delta lowering
* deterministic replay discipline
* correct Rust vs Julia responsibilities
* absence of dependency inversions

If uncertain, **do not add functionality**.

---

### BEGIN.

 

