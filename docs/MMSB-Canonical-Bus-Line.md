| **Bus**          | **Primary Role**            | **Can Write to StateBus?** | **Write Type**                        | **Requires Judgment?** | **Notes**                                    |
| ---------------- | --------------------------- | -------------------------- | ------------------------------------- | ---------------------- | -------------------------------------------- |
| **JudgmentBus**  | Authority / decision making | Yes                        | Approved actions (A)                  | N/A (is judgment)      | Sole source of permission and intent         |
| **StateBus**     | Canonical state admission   | Yes (exclusive)            | State mutations (\Delta_s), facts (F) | Only validates proofs  | Single writer to MMSB                        |
| **ExecutionBus** | Mechanical execution        | Yes                        | Facts (F)                             | No                     | Outcomes of already-approved actions         |
| **LearningBus**  | Derivation / analysis       | Yes                        | Facts (F)                             | No                     | Advisory, non-authoritative                  |
| **ComputeBus**   | Deterministic computation   | Yes                        | Facts (F)                             | No                     | Pure function outputs                        |
| **ChromiumBus**  | External I/O observation    | Yes                        | Facts (F)                             | No                     | World-facing effects, sealed as observations |
| **ReplayBus**    | Observability / audit       | Yes (optional)             | Facts (F)                             | No                     | Historical truth, no causality               |


### Variables

Let

* ( J ) = JudgmentBus
* ( S ) = StateBus
* ( M ) = MMSB
* ( E ) = ExecutionBus
* ( L ) = LearningBus
* ( C ) = ComputeBus
* ( K ) = ChromiumBus
* ( R ) = ReplayBus

Let

* ( A ) = Approved Action
* ( F ) = Fact / Outcome
* ( \Delta_s ) = Canonical State Mutation

---

### Latent Equations (Reference)

[
A : J \rightarrow S \rightarrow M
]

[
F : (E,L,C,K,R) \rightarrow S \rightarrow M
]

[
F \Rightarrow \Delta_s ;\Longrightarrow; F \rightarrow J \rightarrow S
]

---

### Bus Responsibility Table

| **Bus**          | **Primary Role**            | **Can Write to StateBus?** | **Write Type**                        | **Requires Judgment?** | **Notes**                                    |
| ---------------- | --------------------------- | -------------------------- | ------------------------------------- | ---------------------- | -------------------------------------------- |
| **JudgmentBus**  | Authority / decision making | Yes                        | Approved actions (A)                  | N/A (is judgment)      | Sole source of permission and intent         |
| **StateBus**     | Canonical state admission   | Yes (exclusive)            | State mutations (\Delta_s), facts (F) | Only validates proofs  | Single writer to MMSB                        |
| **ExecutionBus** | Mechanical execution        | Yes                        | Facts (F)                             | No                     | Outcomes of already-approved actions         |
| **LearningBus**  | Derivation / analysis       | Yes                        | Facts (F)                             | No                     | Advisory, non-authoritative                  |
| **ComputeBus**   | Deterministic computation   | Yes                        | Facts (F)                             | No                     | Pure function outputs                        |
| **ChromiumBus**  | External I/O observation    | Yes                        | Facts (F)                             | No                     | World-facing effects, sealed as observations |
| **ReplayBus**    | Observability / audit       | Yes (optional)             | Facts (F)                             | No                     | Historical truth, no causality               |

---

### English Explanation

* **Only StateBus writes MMSB** — this invariant never breaks.
* **JudgmentBus writes intent**, not outcomes.
* **All other buses may write facts directly to StateBus** because facts contain **no choice**.
* The moment a fact implies a **new decision or structural change**, it must be routed back through JudgmentBus.

This preserves:

* authority isolation
* mechanical throughput
* deterministic replay
* zero ambiguity over “who decides vs who observes”

---

### Optimality Statement

[
\max(\text{Intelligence}, \text{Efficiency}, \text{Correctness}, \text{Alignment}, \text{Determinism}) = \textbf{Good}
]

This table encodes the maximal-good architecture.
