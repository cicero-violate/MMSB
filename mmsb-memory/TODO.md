Here is a **concrete, ordered, non-hand-wavy TODO list** specifically for **cleanly moving pieces out of `mmsb-memory` and locking its role**.

This is written so you (or Claude) can execute it step-by-step **without architectural drift**.

---

# ✅ MMSB Memory Refactor — Canonical TODO List

## Goal (lock this mentally)

> `mmsb-memory` = **truth semantics only**
> No hardware, no scheduling, no runtime, no execution substrate

---

## **TODO 1 — Freeze the semantic contract of mmsb-memory**

**Before moving anything**

* Declare (in a short README or module comment):

  ```
  mmsb-memory owns:
  - canonical truth
  - commit semantics
  - invariant enforcement
  - deterministic replay semantics
  - proof production (D / E / F)
  ```

* Explicitly state:

  * memory does **not** own performance
  * memory does **not** own hardware
  * memory does **not** own scheduling

This prevents accidental re-imports later.

---

## **TODO 2 — Rename and re-home `module.rs`**

**Low risk, high clarity**

* Rename:

  ```
  mmsb-memory/src/module.rs
  → mmsb-memory/src/memory_engine.rs
  ```

* Rename type:

  ```
  MemoryModule → MemoryEngine
  ```

* Ensure:

  * no lifecycle language
  * no “module” wording anywhere
  * constructor + methods only

✅ Result: memory is clearly an **engine**, not a runtime participant.

---

## **TODO 3 — Identify and tag execution-substrate folders**

**Do not move yet — mark first**

Tag these directories as **execution substrate**:

```
physical/
device/
propagation/
optimization/
```

Add a `README.md` or comment at their root:

> “Execution substrate — to be moved to mmsb-executor”

This prevents accidental new dependencies.

---

## **TODO 4 — Extract execution substrate into `mmsb-executor`**

**Main structural move**

Create a new crate (or reuse existing):

```
mmsb-executor/
```

Move the following directories **verbatim** (no refactors yet):

| From `mmsb-memory` | To `mmsb-executor` |
| ------------------ | ------------------ |
| `physical/`        | `physical/`        |
| `device/`          | `device/`          |
| `propagation/`     | `propagation/`     |
| `optimization/`    | `optimization/`    |

Do **not** change internal code yet.
This is a **pure relocation**.

---

## **TODO 5 — Split “what” vs “how” APIs**

**Critical semantic boundary**

In `mmsb-memory`:

* Keep **interfaces that describe requirements**, e.g.:

  * propagation plan
  * commit intent
  * materialization description

In `mmsb-executor`:

* Implement:

  * GPU kernels
  * buffers
  * queues
  * schedulers
  * fast paths

Rule:

> Memory describes **what must happen**.
> Executor decides **how it happens**.

---

## **TODO 6 — Purge allocation & synchronization from memory**

Specifically audit and move:

From `page/` and `delta/`:

* `allocator.rs`
* `lockfree_allocator.rs`
* `host_device_sync.rs`
* SIMD / buffer management

Keep in memory only:

* logical page identifiers
* page metadata
* invariant checks
* replay validation

---

## **TODO 7 — Lock replay semantics inside memory**

Replay logic **stays**, but must be pure:

In `replay/`:

* Ensure replay:

  * takes sealed events as input
  * produces deterministic outcomes
  * does not emit events itself
  * does not touch storage directly

Add invariant:

> replay(memory, history) → same state

---

## **TODO 8 — Ensure memory has ZERO runtime imports**

Run a dependency audit:

`mmsb-memory` must **not** import:

* tokio
* async runtimes
* threading primitives
* filesystem APIs
* OS APIs

Allowed:

* `mmsb-proof`
* `mmsb-authenticate`
* `mmsb-events` (EventSink only)
* pure std types

---

## **TODO 9 — Add a “read-only memory view” for learning**

To prevent back-edges:

* Define a read-only facade:

  ```
  MemoryView
  ```
* Expose only:

  * outcomes
  * proof references
  * replay snapshots

Ensure `mmsb-learning` depends on **MemoryView**, not full memory.

---

## **TODO 10 — Final invariant check**

Before considering this done, verify:

* `mmsb-memory` can:

  * be instantiated in a test
  * replay history offline
  * verify invariants
  * produce proofs
* **without**:

  * executor
  * service
  * storage

If that holds → memory is correct.

---

# 🔒 Final Lock Statement (keep this visible)

> **mmsb-memory defines truth.
> mmsb-executor realizes execution.
> mmsb-service wires time.
> mmsb-storage persists facts.**

If you want, next we can:

* produce the **mirror TODO list for `mmsb-executor`**
* write the **minimal `mmsb-executor` crate skeleton**
* or do the **same refactor plan for `mmsb-judgment`**

