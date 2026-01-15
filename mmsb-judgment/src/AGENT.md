# AGENT — Judgment Boundary Contract

This document defines the **contract for any agent** (human, AI, or hybrid)
that interacts with the judgment boundary in MMSB.

It does **not** define how judgment is made.
It defines how judgment may be **expressed** to the system.

---

## Core Principle

> **No irreversible state change occurs without explicit judgment.**

Judgment is external to MMSB core.
Agents may participate in judgment **only by issuing permission**, never by bypassing it.

---

## Current Mode (Locked)

### Model A — Human-Issued Judgment

**Status:** Active  
**Description:**  
All judgment is issued explicitly by a human outside the system.

**Properties:**
- Judgment is conscious, explicit, and synchronous
- Each judgment produces exactly one `JudgmentToken`
- Tokens are single-use and non-transferable
- The system enforces judgment structurally
- No background loops, no auto-issue, one judgment per invocation

**Agent Role (Current):**
- Agents may assist humans
- Agents may analyze intent
- Agents may recommend actions
- Agents may **not** issue judgment

---

## Agent Capabilities (Allowed)

An agent **may**:

- Observe intent metadata
- Analyze potential impact
- Classify intent (non-binding)
- Recommend whether judgment should be granted
- Assist a human in decision-making
- Operate entirely outside MMSB core

An agent **may not**:

- Issue `JudgmentToken` in the current model
- Commit state directly
- Bypass the judgment gate
- Store or replay judgment tokens
- Convert analysis into execution

---

## Future Upgrade Paths (Deferred)

The following upgrades are **explicitly deferred**.
They are documented to prevent accidental or implicit adoption.

### P1 — Model C: Selective Delegation

**Description:**  
Allow certain classes of intent to be judged by a delegated agent.

**Requirements:**
- Delegation must be explicit
- Delegation must be revocable
- Delegated judgment must still issue a `JudgmentToken`
- The judgment gate remains unchanged

**Non-Goals:**
- No silent delegation
- No default auto-approval

---

### P2 — Judgment Classification

**Description:**  
Introduce a formal taxonomy of intent to guide who may judge.

**Examples:**
- Reversible vs irreversible
- Local vs global impact
- Low-risk vs high-risk

**Constraints:**
- Classification informs *authority*, not *execution*
- Misclassification must fail safe (escalate to human)

---

### P3 — Audit Trails (Outside Judgment)

**Description:**  
Record that judgment occurred, without encoding why.

**Rules:**
- Audit must not affect execution
- Audit must not be replayable
- Audit must not auto-approve future actions

Audit is observational, never causal.

---

### P4 — Formal Proof Layer

**Description:**  
Formally prove that no commit path exists without a `JudgmentToken`.

**Scope:**
- Structural invariant only
- No proof of “correct judgment”
- No encoding of morality or policy

---

## Boundary Note (Why `request_commit` Is Crate-Only)

The commit boundary remains a **law**, not a convenience:

- `request_commit` stays `pub(crate)` so only in-crate orchestrators can call it.
- External callers (FFI, tests, tools) must pass through judgment flow, never call the commit boundary directly.
- This prevents authority creep and preserves judgment outside execution.

---

## Explicit Prohibitions (Permanent)

The following are **never allowed**, even in future models:

- Autonomous judgment inside MMSB core
- Configurable judgment thresholds
- Heuristic-based auto-approval
- Learning systems that bypass explicit permission
- Retroactive justification of commits

If any of these appear, the system is considered compromised.

---

## Summary

- Judgment is not intelligence
- Judgment is not optimization
- Judgment is not speed
- Judgment is not learning

Judgment is **choice under irreversibility**.

Agents may assist judgment.
Agents may never replace it without explicit, structural delegation.

---

## Final Invariant

> **If an agent can cause reality to change, it must do so only by consuming an explicitly issued JudgmentToken.**
