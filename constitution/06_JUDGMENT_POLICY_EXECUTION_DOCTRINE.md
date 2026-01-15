# JUDGMENT POLICY EXECUTION DOCTRINE

This doctrine is the semantic keystone for the MMSB system.
It freezes the meaning of Judgment, Policy, and Execution.
Any drift here is a constitutional violation.

---
## Section 1 — Definitions (Non-negotiable)

### Judgment
- A human act of choice under uncertainty
- Scarce, external, interruptive
- Cannot be optimized, replayed, or delegated
- Issues authorization only

### Policy
- Law-like constraints
- Timeless, inspectable, deterministic
- Defines may / may not
- Never decides should

### Execution
- Mechanical state transition
- Deterministic
- Fast
- Blind to intent

Formal statement:
J = choice, P = law, E = mechanics

---
## Section 2 — Allowed Dependency Graph

Judgment --issues--> Token
                        |
                        v
                  Policy Check
                        |
                        v
                    Execution

Explicit prohibitions:
- Execution MUST NOT call Judgment
- Policy MUST NOT issue Judgment
- Judgment MUST NOT perform Execution

---
## Section 3 — Hard Invariants (Tie to Tests)

- No execution without JudgmentToken
- Tokens are:
  - non-cloneable
  - non-constructible in core
  - single-use
- Replay with judgment is deterministic

References:
- mmsb-core/tests/compile_fail/*
- mmsb-core/tests/judgment_integration.rs
- mmsb-core/tests/judgment_replay.rs

---
## Section 4 — Forbidden Patterns (Constitutional Violations)

The following patterns are forbidden:
- Auto-judgment
- Background judgment
- Policy-based judgment
- Executor-issued judgment
- LLM-issued judgment

Any appearance of these patterns is a constitutional violation, not a bug.

---
## Section 5 — Change Control

Any change that touches Judgment, Policy, or Execution boundaries:
- Requires constitutional review
- Requires new tests
- Must not be merged silently

