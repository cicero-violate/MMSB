# MMSB GC & FFI Safety Specification

This document defines the memory-safety rules for MMSB when interacting with Rust via FFI.
It describes invariants, when to apply `GC.@preserve`, correct usage patterns, and dangerous cases that must be avoided.

---

## 1. Core Principles

### 1.1 Julia objects may move during GC
If Rust reads or writes into memory owned by Julia (e.g. a `Vector`), the underlying buffer may move during GC unless explicitly preserved.

### 1.2 Rust handles are stable; Julia memory is not
Rust structs representing Pages, Deltas, Allocators are stable pointers. Julia *does not own* this memory.

### 1.3 Only pointers derived from Julia objects require `GC.@preserve`
For example: `pointer(buf)` **must** be preserved.
Pure handles (e.g. `RustPageHandle`) do **not** need preservation.

---

## 2. When to Use `GC.@preserve`

### 2.1 ALWAYS preserve when passing a Julia pointer into Rust
Examples:
```julia
buf = Vector{UInt8}(undef, n)
GC.@preserve buf begin
    ccall(..., pointer(buf), ...)
end
```

### 2.2 ALWAYS preserve any Julia object that owns the target buffer
If Rust reads fields from a Page, Delta, or State struct *allocated on Julia's heap*, preserve the wrapper object.

### 2.3 No preservation required for scalars or Rust-owned memory
Correct:
```julia
FFIWrapper.rust_allocator_release!(handle, UInt64(id))
```

---

## 3. System-wide Invariants

### 3.1 Rust never stores borrowed Julia pointers
All FFI boundaries must avoid long-lived references to Julia memory.

### 3.2 Rust may return pointer/length for read-only buffers
These must be copied immediately using `unsafe_wrap`, `unsafe_load`, or similar.

### 3.3 No Rust → Julia callbacks without explicit lifetime audits
Callbacks introduce complex reentrancy and are prohibited unless carefully audited.

---

## 4. Module Safety Requirements

### 4.1 FFIWrapper.jl
All ctors/getters that copy memory from Rust into Julia *must* preserve destination buffers.
All FFI calls taking Julia pointers must be wrapped in `GC.@preserve`.

### 4.2 ReplayEngine.jl
Replay must preserve:
- The source state,
- All deltas being replayed,
- All pages accessed during clone or hydrate.

### 4.3 DeltaRouter.jl
`route_delta!` must preserve both the delta and the page containing the target handle.

### 4.4 Page.jl
`read_page` uses:
```julia
GC.@preserve page buf begin
    ccall(... pointer(buf) ...)
end
```
This is mandatory.

---

## 5. Fuzz & Stress Guarantees

The following must remain crash-free:

- GC stress during delta routing
- GC stress during propagation
- GC stress during replay
- GC stress during checkpoint/load

Randomized fuzz tests validate invariants but do not assert semantic consistency.

---

## 6. Forbidden Practices

### ❌ Passing `pointer(x)` to Rust without `GC.@preserve`
### ❌ Returning raw Julia pointers to Rust for long-term storage
### ❌ Allowing Rust to assume Julia memory stability
### ❌ Using `unsafe_wrap` without copying data that Rust may later free

---

## 7. Summary

MMSB is memory-safe when the following are true:

1. Every FFI call that passes Julia memory uses `GC.@preserve`.
2. Every Rust buffer is copied into Julia immediately.
3. Rust does not hold borrowed Julia pointers.
4. Stress and fuzz tests validate GC correctness.

