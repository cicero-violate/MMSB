# Performance Investigation Summary

**Date:** 2025-12-17  
**Execution:** Profiling completed via `detailed_profile.jl`

---

## Variables

$$T_{\text{alloc}} = 6.09 \text{ μs}$$ = measured allocation time
$$T_{\text{prop}} = 169.50 \text{ μs}$$ = measured propagation time  
$$T_{\text{recompute}} \approx 80 \text{ μs}$$ = recompute function overhead
$$T_{\text{target\_alloc}} = 1 \text{ μs}$$ = allocation target
$$T_{\text{target\_prop}} = 10 \text{ μs}$$ = propagation target

---

## Key Equations

### Overhead Ratios

$$R_{\text{alloc}} = \frac{T_{\text{alloc}}}{T_{\text{target\_alloc}}} = \frac{6.09}{1} = 6.09$$

$$R_{\text{prop}} = \frac{T_{\text{prop}}}{T_{\text{target\_prop}}} = \frac{169.50}{10} = 16.95$$

### Propagation Decomposition

$$T_{\text{prop}} = T_{\text{diff}} + T_{\text{delta}} + T_{\text{route}} + T_{\text{propagate}}$$

Where:
- $T_{\text{diff}} \approx 8 \text{ μs}$ (diff computation)
- $T_{\text{delta}} \approx 15 \text{ μs}$ (delta creation)
- $T_{\text{route}} \approx 25 \text{ μs}$ (routing without propagation)
- $T_{\text{propagate}} \approx 120 \text{ μs}$ (propagation cascade)

### Recompute Dominance

$$\frac{T_{\text{recompute}}}{T_{\text{prop}}} = \frac{80}{169.50} \approx 0.47 = 47\%$$

### GPU Overhead Ratio

$$R_{\text{GPU}} = \frac{T_{\text{GPU}}}{T_{\text{CPU}}} = \frac{3900}{22.4} \approx 174$$

For 1 KB pages, GPU is 174× slower than CPU.

---

## Critical Findings

### 1. Allocation: 6× over target

**Root cause:** FFI overhead in `rust_tlog_new()` + `rust_allocator_new()`

**Solution:** State pooling
$$T_{\text{alloc\_optimized}} = T_{\text{pool\_acquire}} + T_{\text{page\_create}} \approx 2-3 \text{ μs}$$

### 2. Propagation: 17× over target

**Root cause:** Unnecessary recomputation (47% of time)

**Solution:** Epoch-based caching
$$T_{\text{prop\_cached}} = T_{\text{prop}} - T_{\text{recompute\_avoided}} \approx 169.50 - 80 = 89.50 \text{ μs}$$

With additional optimizations (lock-free IDs, batch events):
$$T_{\text{prop\_optimized}} \approx 60 \text{ μs}$$

### 3. GPU: 174× slower for small pages

**Root cause:** Transfer overhead dominates

**Solution:** Threshold-based routing
$$\text{use\_GPU} = \begin{cases} 
\text{true} & \text{if } \text{page\_size} \geq 1 \text{ MB} \\
\text{false} & \text{otherwise}
\end{cases}$$

### 4. Batch scaling degrades

$$\frac{T_{\text{batch128}}}{128} = 151 \text{ μs/delta} > \frac{T_{\text{batch32}}}{32} = 125 \text{ μs/delta}$$

Efficiency drops by 21% at large batch sizes.

**Solution:** Parallel routing with thread pool

---

## Implementation Priority

### Phase 1 Impact

$$\Delta T_{\text{prop}} = 169.50 - 60 = 109.50 \text{ μs saved}$$
$$\text{Improvement} = \frac{109.50}{169.50} \times 100\% = 64.6\%$$

### Total Expected Improvement

| Metric | Before | After Phase 2 | Reduction |
|--------|--------|---------------|-----------|
| Allocation | 6.09 μs | 2 μs | 67% |
| Propagation | 169.50 μs | 40 μs | 76% |

---

## Conclusion

Primary bottleneck identified: **Recompute function executes unnecessarily** (80 μs, 47% of propagation time).

Root cause: No epoch tracking to detect unchanged parent pages.

Solution ROI: Implementing cache-aware propagation yields 65% latency reduction for propagation workloads.

---

## Risk Equations & Invariants

### Recompute Purity Constraint

$$\forall t: \text{recompute}(page, t) = f(\text{parent\_state})$$

Violation condition:
$$\exists t_1, t_2: \text{parent\_state}(t_1) = \text{parent\_state}(t_2) \land \text{recompute}(page, t_1) \neq \text{recompute}(page, t_2)$$

### Atomic Contention Model

False sharing overhead:
$$L_{\text{atomic}} = L_{\text{base}} + k \cdot N_{\text{threads}} \cdot P_{\text{collision}}$$

Per-thread range amortization:
$$L_{\text{range}} = \frac{L_{\text{global}}}{R} + L_{\text{local}}$$

where $R$ = range size, $L_{\text{local}} \ll L_{\text{global}}$.

### State Pool Determinism

$$\text{hash}(\text{reset!}(s_{\text{pooled}})) = \text{hash}(s_{\text{fresh}})$$

Non-determinism detection:
$$\Delta_{\text{replay}} = |\text{epoch}_{\text{fresh}} - \text{epoch}_{\text{pooled}}| > 0 \Rightarrow \text{FAILURE}$$

### Delta Commutativity

For same page $p$:
$$\Delta_i(p) \circ \Delta_j(p) \neq \Delta_j(p) \circ \Delta_i(p) \text{ if } \text{epoch}_i \neq \text{epoch}_j$$

Parallelization constraint:
$$\text{parallel}(\{\Delta_i\}) \Leftrightarrow \forall i,j: \text{page}(\Delta_i) \neq \text{page}(\Delta_j)$$

### GPU Overhead Decomposition

$$G = G_{\text{launch}} + G_{\text{transfer}} + G_{\text{sync}}$$

$$G_{\text{transfer}} = \frac{\text{size}}{\text{bandwidth}} + L_{\text{PCIe}}$$

Threshold condition:
$$\text{size} > \theta \Leftrightarrow G < T_{\text{CPU}}$$

where $\theta \approx 1 \text{ MB}$ empirically.

### Batch Scaling Overhead

$$B(n) = \frac{C_{\text{group}}(n) + C_{\text{route}}(n)}{n}$$

$$C_{\text{group}}(n) = O(n \log n) \text{ (sorting + grouping)}$$

Efficiency degradation:
$$\frac{dB}{dn} > 0 \text{ for } n > n_{\text{threshold}} \Rightarrow \text{sequential bottleneck}$$

### Overall Correctness

$$\text{Good} = \bigwedge \{\text{Purity}, \text{Determinism}, \text{Commutativity}, \text{Ordering}\}$$

Any single violation:
$$\neg \text{Good} \Rightarrow \text{Replay failure (existential risk)}$$
