# Phase 5: CLAUDE.md Architectural Compliance

## Executive Summary

**Status**: Architecture 95% compliant with CLAUDE.md specification
**Action Required**: Implement 5 missing critical components
**Timeline**: 6 weeks (Weeks 21-26)
**Effort**: 140 hours

---

## Feasibility Assessment: ✅ FEASIBLE

The CLAUDE.md architectural specification is **fully achievable** with the current MMSB codebase. The core infrastructure (13 layers, 702 code elements) is correctly structured. Missing components are **additive** rather than requiring architectural redesign.

### What's Already Correct ✅

1. **Layer Boundaries**: 13-layer architecture matches spec exactly
2. **Rust/Julia Separation**: Rust dominates L0-L5 (execution), Julia dominates L7-L12 (reasoning)
3. **Delta System**: Layer 1 has 120 elements (81 Rust + 39 Julia) for page/delta management
4. **Replay Engine**: Deterministic replay exists in `ReplayEngine.jl`
5. **TLog Infrastructure**: Transaction logging complete
6. **No Learning in Core**: L0-L5 are pure execution (no neural backprop)

---

## Critical Gaps Identified

### Gap 1: UpsertPlan Structure 🔴 CRITICAL

**What's Missing**: No first-class `UpsertPlan` structure in Layer 7

**CLAUDE.md Requirement**:
```julia
UpsertPlan = {
    Query: State → Selection
    Predicate: Selection → Bool
    DeltaSpec: Selection → Δ*
    Metadata: Intent context
}
```

**Current State**: L7 has `Intention`, `Goal`, `IntentionState` but no `UpsertPlan`

**Solution**: Create `src/07_intention/UpsertPlan.jl`

**Effort**: 8 hours

---

### Gap 2: Intent → Delta Lowering Pipeline 🔴 CRITICAL

**What's Missing**: No explicit lowering from L7 (Intent) to L1 (Delta)

**CLAUDE.md Requirement**:
```
Intent → UpsertPlan → DeltaSpec → Validation → TLog → Propagation
```

**Current State**: Intent formation exists, delta creation exists, but no connecting lowering function

**Solution**: 
- `src/07_intention/intent_lowering.jl` (Julia - generates delta specs)
- FFI bridge to pass delta specs to Rust validation

**Effort**: 56 hours (lowering 24h + FFI 32h)

---

### Gap 3: Intent Metadata in TLog 🟡 MODERATE

**What's Missing**: TLog doesn't persist intent metadata for replay

**CLAUDE.md Requirement**: Intent must be replayable from TLog

**Current State**: TLog stores deltas but no intent context

**Solution**: Extend TLog entry format with optional `intent_metadata` field

**Effort**: 16 hours

---

### Gap 4: Delta Validation Separation 🟡 MODERATE

**What's Missing**: Validation not separated from delta application

**CLAUDE.md Requirement**: Delta validation happens in Rust (L1) before execution

**Current State**: `Delta::apply_to()` applies deltas, unclear if separate validation exists

**Solution**: Extract validation into `src/01_page/delta_validation.rs`

**Effort**: 12 hours

---

### Gap 5: QMU API Clarity 🟢 LOW

**What's Missing**: Query/Mutate/Upsert operations not explicitly labeled

**CLAUDE.md Requirement**: Clear QMU model

**Current State**: `API.jl` has `query_page`, `update_page` but not documented as QMU

**Solution**: Add documentation clarifying QMU boundaries

**Effort**: 8 hours

---

## Implementation Plan

### Week 21: Foundation (24 hours)
- Define UpsertPlan structure
- Extend TLog metadata schema

### Week 22: Core Logic (36 hours)
- Implement intent lowering in Julia
- Separate delta validation in Rust

### Week 23: Integration (32 hours)
- Create FFI bridge for lowering
- Test Julia ↔ Rust deltaspec transfer

### Week 24: Testing (24 hours)
- End-to-end intent → execution tests
- QMU separation verification

### Week 25: Replay (16 hours)
- Test replay with intent metadata
- Verify intent reconstruction

### Week 26: Documentation (8 hours)
- Document QMU API
- Create CLAUDE.md compliance report

---

## Success Metrics

Phase 5 is complete when:

- ✓ `UpsertPlan` struct exists in `src/07_intention/UpsertPlan.jl`
- ✓ Intent lowering produces valid delta specs
- ✓ TLog persists and replays intent metadata
- ✓ Delta validation prevents invalid operations
- ✓ All 10 CLAUDE.md non-negotiable rules verified:
  1. Rust executes, never reasons ✓
  2. Julia reasons, never mutates directly ✓
  3. All state changes via canonical deltas ✓
  4. Intent ≠ execution (intent persisted) ✓
  5. Deterministic replay works ✓
  6. No learning in L0-L5 ✓
  7. No cognition below L6 ✓
  8. No layer boundary violations ✓
  9. No abstraction collapse ✓
  10. UpsertPlan first-class and functional ✓

---

## Risk Assessment

### Low Risk ✅
- UpsertPlan structure (simple Julia struct)
- Delta validation (straightforward Rust refactor)
- QMU documentation (no code changes)

### Medium Risk ⚠️
- TLog schema extension (backward compatibility required)
- Intent lowering logic (complex type conversions)

### High Risk 🔴
- FFI bridge (memory safety, performance overhead)
- End-to-end integration (many moving parts)

**Mitigation**: Incremental testing at each step, extensive FFI safety checks

---

## Dependency Graph

```
Week 21: [UpsertPlan] + [TLog Schema]
           |              |
           v              v
Week 22: [Intent Lowering] + [Delta Validation]
                  |                   |
                  +--------+----------+
                           v
Week 23:          [FFI Bridge]
                           v
Week 24:     [Integration Tests]
                           v
         +----------------+----------------+
         v                                 v
Week 25: [Replay Tests]          Week 26: [QMU Docs]
```

---

## Files to Create (4 new files)

1. `src/07_intention/UpsertPlan.jl` - UpsertPlan structure
2. `src/07_intention/intent_lowering.jl` - Lowering pipeline
3. `src/01_page/delta_validation.rs` - Delta validator
4. `test/test_claude_compliance.jl` - Compliance test suite

## Files to Modify (8 existing files)

1. `src/01_page/tlog.rs` - Add intent metadata field
2. `src/01_page/TLog.jl` - Handle intent metadata
3. `src/01_page/delta.rs` - Use validation before apply
4. `src/ffi.rs` - Add lowering FFI functions
5. `src/ffi/FFIWrapper.jl` - Add Julia wrappers
6. `src/API.jl` - Document QMU operations
7. `src/07_intention/IntentionTypes.jl` - Integrate UpsertPlan
8. `test/test_layer07_intention.jl` - Add UpsertPlan tests

---

## Deliverables

- **Code**: 12 files added/modified
- **Tests**: ~20 new test cases
- **Documentation**: QMU API guide + compliance report
- **Verification**: All CLAUDE.md rules validated

---

## Next Steps

1. **Review this plan** with team
2. **Prioritize tasks** if time-constrained
3. **Begin Week 21** with UpsertPlan definition
4. **Track progress** in `06_TASK_LOG_PHASE_5.md`

---

## Questions for Clarification

Before starting implementation, confirm:

1. **TLog Format**: Should we version the TLog format or maintain full backward compatibility?
2. **FFI Performance**: What's acceptable overhead for intent lowering? <1ms? <10ms?
3. **Intent Replay**: Should replay reconstruct full intent or just metadata?
4. **Validation Strictness**: Should invalid deltas be rejected silently or raise errors?
5. **Priority**: Is CLAUDE.md compliance blocking other features?
