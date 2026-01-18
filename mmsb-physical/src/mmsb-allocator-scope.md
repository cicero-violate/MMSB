mmsb-physical/
├── allocator_stats.rs
├── memory_pool.rs        // raw ptr + size
├── arena.rs              // optional
├── slab.rs               // optional
└── lib.rs / mod.rs


## Ground invariant (re-applied)

> **`mmsb-allocator` may depend ONLY on `mmsb-primitives`.**
> If a file mentions `Page`, `Delta`, `Epoch`, `Device`, or semantics → it does **not** belong there.

---

## File-by-file verdict

### 1. `PageAllocator.rs`

**❌ MUST NOT be in `mmsb-allocator`**

Reason:

* It is page-semantic
* It reasons about `Page`, `PageLocation`, epochs, metadata
* It enforces invariants

**Correct home**

```
mmsb-memory/src/page/page_allocator.rs
```

**Rule**
[
\text{PageAllocator} \in \text{memory},; \text{never allocator}
]

Allocator provides *resources*, not *pages*.

---

### 2. `lockfree_allocator.rs`

**❌ ALSO does NOT belong in `mmsb-allocator` (as written)**

Because:

* It operates on `Page`
* It assumes page layout / semantics

Two valid outcomes:

#### Option A (most likely correct)

Move to:

```
mmsb-memory/src/page/lockfree_page_allocator.rs
```

#### Option B (only if rewritten)

It may stay in `mmsb-allocator` **only if** rewritten to:

* operate on raw pointers
* operate on `(id, size)`
* know nothing about `Page`, `Epoch`, `Location`

Right now, it does **not** qualify.

---

### 3. `simd_mask.rs`

**❌ Definitely NOT allocator**

This is:

* vectorized compute
* execution / device-adjacent
* data-parallel logic

**Correct home**

```
mmsb-device/src/simd/simd_mask.rs
```

or

```
mmsb-physical/src/simd_mask.rs
```

Allocator must not contain SIMD, ever.

---

### 4. `mod.rs`

This file follows whatever survives in the crate.

After cleanup, `mmsb-allocator/src/mod.rs` should export **only**:

```rust
pub mod allocator_stats;
pub mod memory_pool;   // if raw
pub mod arena;         // if raw
```

If `mod.rs` currently re-exports any of:

* `PageAllocator`
* `Page`
* `Device*`
* `SIMD`

then it is **wrong by definition**.

---

## What `mmsb-allocator` should look like after fixes

```
mmsb-allocator/
├── allocator_stats.rs
├── memory_pool.rs        // raw ptr + size
├── arena.rs              // optional
├── slab.rs               // optional
└── lib.rs / mod.rs
```

Allowed imports:

```rust
use std::alloc::*;
use std::ptr::*;
use std::sync::atomic::*;
use mmsb_primitives::PageID; // opaque only
```

Nothing else.

---

## One-line kill rule (keep this)

> **If a file mentions `Page`, it is not an allocator.**

Apply this mechanically and the architecture snaps into place.



