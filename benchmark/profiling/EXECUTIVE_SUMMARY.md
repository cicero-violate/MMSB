# MMSB Performance Investigation - Executive Summary

**Date:** 2025-12-17  
**Status:** Investigation Complete  
**Priority:** P1 (Critical path blocker)

---

## Key Findings

### Measured Performance

| Metric | Measured | Target | Ratio |
|--------|----------|--------|-------|
| Allocation | 6.09 μs | 1 μs | 6× |
| Propagation | 169.50 μs | 10 μs | 17× |

### Root Causes

1. **Propagation (80 μs, 47%):** Unnecessary recomputation - no epoch tracking
2. **Allocation (6 μs):** FFI overhead in `rust_tlog_new()` + `rust_allocator_new()`
3. **GPU (174× slower):** Transfer overhead dominates for <1MB pages
4. **Batch (21% degradation):** Sequential bottleneck at large sizes

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)

**Targets:** Propagation 169 → 60 μs (65% reduction)

1. Cache-aware propagation with epoch tracking
2. GPU size threshold (1 MB)
3. Atomic delta ID allocation

### Phase 2: Architecture (2-3 days)

**Targets:** Allocation 6 → 2 μs, Propagation 60 → 40 μs

1. State pooling with deterministic reset
2. Per-thread delta ID ranges
3. Parallel batch routing

---

## Critical Risks & Mitigations

### Risk 1: Recompute Purity (CRITICAL)

**Issue:** Epoch cache breaks if recompute has side-effects or non-parent deps.

**Mitigation:** Add recompute dependency signature validation.

### Risk 2: State Pool Leakage (CRITICAL)

**Issue:** Residual state causes non-deterministic replay.

**Mitigation:** Mandatory `reset!()` with hash equality test vs fresh state.

### Risk 3: Delta Ordering

**Issue:** Parallel routing must preserve per-page epoch ordering.

**Mitigation:** Parallelize across pages, serialize within page.

---

## Success Criteria

| Phase | Allocation | Propagation | Status |
|-------|-----------|-------------|--------|
| Baseline | 6.09 μs | 169.50 μs | Current |
| Phase 1 | 6 μs | 60 μs | Architectural target |
| Phase 2 | 2 μs | 40 μs | Architectural target |
| Ultimate | 1 μs | 10 μs | Requires algorithm changes |

**Correctness invariant:** All optimizations must preserve deterministic replay.

---

## Artifacts Generated

- `INVESTIGATION_REPORT.md` - Detailed analysis + roadmap + risk mitigations
- `ANALYSIS_SUMMARY.md` - Mathematical breakdown + risk equations
- `detailed_profile.jl` - Measurement script (6.09 μs / 169.50 μs confirmed)

---

## Recommendation

**Proceed with Phase 1 implementation.** Expected 65% propagation latency reduction with manageable risk if correctness constraints are enforced.
