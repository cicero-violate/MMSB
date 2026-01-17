## Core Authority & Truth
mmsb-intent
Produces: A
Owns intent schema + bounds

mmsb-policy
Produces: B
Classification only

mmsb-judgment
Produces: C
Only authority module

mmsb-memory
Produces: D,E,F
Owns truth, time, invariants

## Execution
mmsb-executor
Consumes: C
Produces: execution requests
No proofs of its own

## Learning & Knowledge
mmsb-learning
Consumes: F
Produces: G

mmsb-knowledge
Stores: G
Read-only to judgment

Infrastructure
mmsb-authenticate
Verifies all proofs
No decisions
Shared library

mmsb-storage
Persists: A →  G
Bytes only

mmsb-service
Event bus runtime
Loads modules
Routes events
No authority

4. Replay, Learning, Audit (Explicit)
🔁 Replay

Replay is simply:
(A,B,C,D,E)⇒r reconstruct state

No learning required.
No executor required.
Memory replays commits.

🧠 Learning

Learning never feeds execution directly:
𝐹→ 𝐺 → Judgment (read-only)

Judgment may:
tighten rules
require confirmation
refuse
But learning never acts.

📜 Audit
Audit verifies:
𝐴⊆𝐵⊆𝐶⊆𝐷⊆𝐸⊆𝐹⊆𝐺
A⊆B⊆C⊆D⊆E⊆F⊆G

If any link is missing → invalid history.
