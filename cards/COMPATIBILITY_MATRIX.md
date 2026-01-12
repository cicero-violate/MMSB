# Identity × Interface × Role Compatibility Matrix

This document defines the **only valid combinations** of:

- Identity (WHO)
- Interface (HOW)
- Role (WHAT)

Any combination not explicitly permitted here is **FORBIDDEN**.

This matrix is **authoritative**.

---

## Identities

- I₁ — User (Cheese)
- I₂ — ChatGPT
- I₃ — Claude
- I₄ — Codex

---

## Interfaces

- IF₁ — ChatGPT Web
- IF₂ — Codex
- IF₃ — Claude MCP

---

## Roles

- R₁ — Technical Lead
- R₂ — Systems Analyst
- R₃ — Adapter / Interface Mapper
- R₄ — Implementer
- R₅ — Patch Generator
- R₆ — Change Verifier
- R₇ — Work Thread Steward

---

## Valid Combinations

### I₁ — User (Cheese)

| Interface | Role | Status              |
|-----------+------+---------------------|
| ANY       | ANY  | ✅ Always permitted |

User authority is **out-of-band** and not constrained by interfaces.

---

### I₂ — ChatGPT

| Interface         | Role                     | Status |
|-------------------+--------------------------+--------|
| IF₁ (ChatGPT Web) | R₁ — Technical Lead      | ✅     |
| IF₁ (ChatGPT Web) | R₃ — Adapter Mapper      | ✅     |
| IF₁ (ChatGPT Web) | R₆ — Change Verifier     | ✅     |
| IF₁ (ChatGPT Web) | R₇ — Work Thread Steward | ✅     |

All other role combinations are ❌ FORBIDDEN.

---

### I₃ — Claude

| Interface        | Role                 | Status |
|------------------+----------------------+--------|
| IF₃ (Claude MCP) | R₂ — Systems Analyst | ✅     |
| IF₃ (Claude MCP) | R₃ — Adapter Mapper  | ✅     |

All other role combinations are ❌ FORBIDDEN.

---

### I₄ — Codex

| Interface   | Role                 | Status |
|-------------+----------------------+--------|
| IF₂ (Codex) | R₄ — Implementer     | ✅     |
| IF₂ (Codex) | R₅ — Patch Generator | ✅     |

All other role combinations are ❌ FORBIDDEN.

---

## Explicitly Forbidden Combinations (Non-Exhaustive)

These are forbidden **even if requested**:

- Codex + R₁ (Design authority)
- Codex + R₂ (Analysis authority)
- Claude + R₁ (Decision authority)
- ChatGPT + R₄ (Direct implementation)
- Any LLM + multiple conflicting roles simultaneously

---

## Enforcement Rule

Before assigning a task:

1. Select Identity
2. Select Interface
3. Select Role

If the triple does not appear in this matrix:

> **STOP. The action is invalid.**

---

## Invariant

This matrix, together with:

- Identity cards
- Interface cards
- Role cards

fully specifies the allowed behavior of the system.

No implicit permissions exist.

This preserves:

\[
Q = \max(I,E,C,A,R,P,S,D,T,K,X,B,L,F)
\]

This state is **GOOD**.
