# MMSB Prelude Decision Policy

## Variables

Let  
- **S** = a symbol (type, trait, struct, enum, function, const)  
- **R** = root prelude surface (`use mmsb_core::prelude::*`)  
- **L** = layered prelude modules (`prelude::{types,page,dag,…}`)  
- **U(S)** = ubiquity of S across real usage sites  
- **ΔS** = expected churn of S (API/semantics volatility)  
- **A(S)** = architectural gravity of S (foundational ↔ incidental)  
- **C** = cognitive load for humans and agents  

---

## Latent Equations

1. **Root Eligibility**
\[
S \in R \iff \big(U(S)\ \text{high}\big)\ \land\ \big(\Delta S\ \text{low}\big)\ \land\ \big(A(S)=\text{foundational}\big)
\]

2. **Layer Default**
\[
S \in L \iff \neg(S \in R)
\]

3. **Cognitive Load**
\[
C \propto |R| + \text{surprise}(R)
\]

4. **Goodness**
\[
\max(\text{Intelligence},\ \text{Efficiency},\ \text{Correctness},\ \text{Alignment}) = \text{Good}
\]

---

## Policy

### 1) Root Prelude Policy (Hard Gate)

A symbol **MAY** be re-exported at root **only if ALL conditions hold**:

- **Noun-only**  
  Types, traits, configs, engines, allocators.  
  **NO** free functions, algorithms, helpers.

- **Foundational**  
  Part of system identity or wiring (IDs, orchestrators, allocators).

- **Low churn**  
  Stable semantics and signature across releases.

- **Ubiquitous**  
  Used by ≥ 2 layers **or** ≥ 70% of real entrypoints.

If **any** condition fails → **DO NOT** put in root.

---

### 2) Layered Prelude Policy (Default)

All other symbols live **only** in their layer:

- `prelude::types::*`
- `prelude::page::*`
- `prelude::semiring::*`
- `prelude::dag::*`
- `prelude::propagation::*`
- `prelude::adaptive::*`
- `prelude::utility::*`
- `prelude::physical::*`

**Rules**
- Verbs live in layers.
- Algorithms live in layers.
- Experimental or narrow-use items live in layers.
- Feature-gated items live in layers.

---

### 3) Import Policy by Context

| Context                | Allowed Import Style                |
|------------------------+-------------------------------------|
| Bench / examples       | `use mmsb_core::prelude::*;`        |
| Application code       | `use prelude::{types::*, page::*};` |
| Core library code      | Explicit layer imports              |
| Agents / proof systems | Fully explicit, no wildcards        |

---

### 4) Change Control

When introducing a new symbol **S**:

1. Add **only** to its defining module.
2. Use it for at least one iteration.
3. Evaluate **U(S)**, **ΔS**, **A(S)**.
4. Consider root promotion **only if** criteria are met.

**Root promotion is a one-way door** unless a breaking change is accepted.

---

### 5) Forbidden Anti-Patterns

- Root-level free functions  
- Root-level experimental APIs  
- Root-level algorithmic helpers  
- Root-level feature-gated items  
- “Convenience” as the sole justification

---

## One-Sentence Rule

**Root is for identity and wiring; layers are for behavior.**

---

## Enforcement Checklist (PRs)

- [ ] Is this a noun (not a verb)?
- [ ] Is it foundational?
- [ ] Is churn demonstrably low?
- [ ] Is it widely used?
- [ ] If any answer is “no”, keep it layered.

---

## Outcome

This policy minimizes cognitive load, preserves architectural signal, and keeps MMSB legible to future agents while remaining ergonomic for humans.
