# Epoch Semantics Specification

## Task ID: T0.1
**Status:** ✅ Formalized  
**Date:** 2025-12-17  
**Author:** DAG Execution Protocol

---

## 1. Definition

### Mathematical Notation

$$\text{Let } E_p(t) = \text{epoch value of page } p \text{ at time } t$$

$$E_p : \mathbb{N} \to \mathbb{N}, \quad E_p(0) = 0$$

### Epoch Semantics

An **epoch** is a per-page monotonic counter that tracks the **content version** of that page.

$$\forall p \in \text{Pages}, \forall t : E_p(t+1) \geq E_p(t)$$

**Increment Rule:**

$$E_p(t+1) = \begin{cases}
E_p(t) + 1 & \text{if delta applied to } p \text{ at time } t \\
E_p(t) & \text{otherwise}
\end{cases}$$

---

## 2. Scope and Invariants

### 2.1 Per-Page Scope

Epochs are **page-local**, not global:

$$E_p \perp E_q \quad \forall p \neq q$$

Each page maintains its own independent epoch counter.

### 2.2 Content Change Indicator

The epoch indicates **when content last changed**:

$$\text{content}(p, t_1) = \text{content}(p, t_2) \iff E_p(t_1) = E_p(t_2)$$

Contrapositive:

$$E_p(t_1) \neq E_p(t_2) \implies \text{content}(p, t_1) \neq \text{content}(p, t_2)$$

### 2.3 Metadata Storage

Epochs are stored in page metadata:

```julia
page.metadata[:epoch_dirty] :: UInt32
```

**Initialization:**

$$E_p(0) = 0 \quad \text{(at page creation)}$$

---

## 3. Delta Application Protocol

### 3.1 Epoch Increment on Apply

When delta $\delta$ is applied to page $p$:

$$\text{apply}(\delta, p) \implies E_p \gets E_p + 1$$

### 3.2 Delta Epoch Field

Each delta carries the **target epoch**:

```julia
struct Delta
    epoch::UInt32  # Expected epoch of target page
    # ...
end
```

**Invariant:**

$$\delta.\text{epoch} = E_{\delta.\text{page\_id}}(\text{creation time of } \delta)$$

### 3.3 Application Semantics

Delta application is **unconditional** at the physical layer:

```julia
function apply_delta!(page, delta)
    # Apply regardless of epoch mismatch
    FFIWrapper.rust_delta_apply!(page.handle, delta.handle)
    page.metadata[:epoch_dirty] += 1
end
```

**Note:** Epoch validation for correctness happens at the **propagation layer**, not the physical layer.

---

## 4. Recompute Dependencies

### 4.1 Dependency Tracking

Pages with recompute functions declare their dependencies:

```julia
page.metadata[:recompute_deps] :: Vector{PageID}
```

### 4.2 Dependency Signature

A **recompute signature** captures parent epochs:

$$S_p = \{(p_i, E_{p_i}) \mid p_i \in \text{deps}(p)\}$$

```julia
struct RecomputeSignature
    parent_ids::Vector{PageID}
    parent_epochs::Vector{UInt32}
end
```

### 4.3 Cache Invalidation Rule

Recompute is required if and only if:

$$\exists i : E_{p_i}(\text{now}) \neq S_p[i].\text{epoch}$$

If all parent epochs match the cached signature, **skip recompute**.

---

## 5. Determinism Guarantees

### 5.1 ID Assignment Order

Epoch increments are **deterministic** given:

1. Deterministic page ID allocation
2. Deterministic delta ID allocation
3. Fixed routing order (no dict iteration)

### 5.2 Replay Invariant

For any checkpoint state $S_0$ and operation sequence $\Omega$:

$$\text{replay}(S_0, \Omega) \implies E_p(\text{final}) = E_p(\text{original final})$$

**Test Oracle:**

```julia
function canonical_snapshot(state)
    Dict(
        "epochs" => Dict(id => p.metadata[:epoch_dirty] 
                        for (id, p) in state.pages)
    )
end
```

### 5.3 Pool Invariant

Fresh state vs pooled state must produce identical epochs:

$$\forall \Omega : E_p^{\text{fresh}}(\Omega) = E_p^{\text{pooled}}(\Omega)$$

---

## 6. Non-Guarantees

Epochs do **NOT** guarantee:

1. **Global ordering** — pages have independent epoch spaces
2. **Unique content** — collisions possible if content cycles
3. **Causality tracking** — use explicit dependency graph for that

---

## 7. Implementation Checklist

### Current Status (T0.1)

- [x] Epoch is per-page `UInt32` counter
- [x] Stored in `page.metadata[:epoch_dirty]`
- [x] Increments on delta application
- [x] Scoped to page lifetime
- [ ] Formal signature comparison (T1.1)
- [ ] Cache invalidation logic (T1.2)
- [ ] Deterministic replay tests (T1.3)

### Files Affected

- `src/01_page/Page.jl` — Page struct metadata
- `src/01_page/Delta.jl` — Delta epoch field
- `src/02_semiring/DeltaRouter.jl` — Epoch increment on apply
- `src/04_propagation/PropagationEngine.jl` — Signature-based recompute (future)

---

## 8. Testing Requirements

### Unit Tests

1. Epoch increments on delta apply
2. Epoch preserved across serialization
3. Independent epoch counters per page

### Integration Tests

1. Deterministic replay produces same epochs
2. Pooled state matches fresh state epochs
3. Signature-based cache invalidation (T1.2)

### Invariant Checks

```julia
@assert page.metadata[:epoch_dirty] isa UInt32
@assert page.metadata[:epoch_dirty] >= 0
```

---

## 9. Performance Impact

**Overhead:**

- Epoch increment: ~1 cycle (metadata write)
- Signature comparison: ~10 ns per dependency

**Benefit:**

- Skip unnecessary recompute: **-80 μs** (T1.2 target)

---

## 10. Future Considerations

### Epoch Overflow

`UInt32` provides $2^{32} = 4.3 \times 10^9$ versions.  
At 1M updates/sec, overflow occurs after **71 minutes**.

**Mitigation:** Use `UInt64` if long-lived pages expected.

### Distributed Systems

Epoch semantics extend to distributed contexts with:

- Vector clocks for causality
- Lamport timestamps for ordering

Not required for single-state MMSB.

---

## 11. References

- **Task:** T0.1 — Define Epoch Semantics
- **Dependency:** None (Level 0 foundation)
- **Blocks:** T1.1, T1.2, T1.3 (Level 1 correctness)
- **Related:** `DAG-to-DO-DETAILED.md` — Project schedule

---

**End of Specification**
