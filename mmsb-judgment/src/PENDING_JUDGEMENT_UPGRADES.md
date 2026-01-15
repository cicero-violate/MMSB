# Pending Judgment-Related Upgrades

This document tracks **deliberately deferred** upgrades to the judgment architecture.
None of the items below are required for correctness of the current system.

The system is currently operating under **Model A — Human-Issued Judgment**.
All items here are optional future evolutions and must preserve the core invariant:

> No irreversible state change occurs without explicit judgment.

---

## P0 — Locked and Complete (Current State)

- Model A: Human-issued judgment
- Structural judgment gate enforced by type system
- Single explicit commit path
- No implicit authority
- No configuration-based bypass
- No delegation

This state is considered **correct, sufficient, and stable**.

---

## P1 — Model C: Selective Delegation (Deferred)

**Description**

Introduce conditional delegation of judgment based on intent classification.

**Concept**

- Some intents may be judged by a delegated agent
- Others require explicit human approval
- Delegation must be explicit, reviewable, and revocable

**Constraints**

- Delegation must live *outside* MMSB core
- Judgment gate remains unchanged
- Delegated judgments still produce `JudgmentToken`
- No silent or default delegation

**Status**

- Deferred
- Not required
- Requires stable intent taxonomy first

---

## P2 — Judgment Classification (Deferred)

**Description**

Formalize intent classes to support selective delegation.

**Examples**

- Reversible vs irreversible
- Local vs global impact
- Low-risk vs high-risk
- Development vs production

**Constraints**

- Classification informs *who may judge*, not *how judgment works*
- Classification must not auto-commit
- Misclassification must fail safe (escalate to human)

**Status**

- Deferred
- Depends on real-world usage patterns

---

## P3 — Audit Trails (Outside Judgment) (Deferred)

**Description**

Record *that* judgment occurred, without encoding *why*.

**Key Principle**

Audit ≠ judgment  
Audit must never substitute for judgment.

**Constraints**

- Audit trails must not affect execution
- Audit data must not be used to auto-approve
- Judgment token remains non-replayable

**Examples**

- Timestamp of approval
- Actor identity (human / agent)
- Intent identifier (opaque)

**Status**

- Deferred
- Optional
- Must not weaken the gate

---

## P4 — Formal Proof Layer Around the Gate (Deferred)

**Description**

Add a formal proof (e.g. Lean-backed or type-level invariant) that:

> It is impossible to reach the commit boundary without a `JudgmentToken`.

**Scope**

- Proof of structural invariant only
- No proof of “correct judgment”
- No encoding of morality or policy

**Constraints**

- Proof layer must not introduce runtime complexity
- Proof must not require judgment logic inside the system

**Status**

- Deferred
- High value, low urgency
- Requires system stability first

---

## Explicit Non-Goals

The following are **intentionally excluded**:

- Fully autonomous judgment
- Configurable judgment thresholds
- Heuristic-based auto-approval
- Learning-driven judgment inside MMSB core
- Performance optimization of judgment

These are considered **misaligned** with the system’s purpose.

---

## Closing Note

Deferred does not mean forgotten.

Each item here represents an **earned promotion**, not a missing feature.
They should only be attempted when the system’s current guarantees feel
boring, obvious, and unquestioned.

Until then, Model A is the reference truth.
