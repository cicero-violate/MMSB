# PHASE 6.5: Static Effect Signature Schema

**Status**: FOUNDATION LOCKED (Version 0.1.0)
**Authority**: CIPT Architectural Law
**Date**: 2026-01-01

---

## Executive Summary

This document defines the **Static Effect Signature Schema**, the keystone of PHASE 6.5: Admission Intelligence Formalization. This schema enables compositional reasoning about transformation action sequences without execution, transitioning the system from a refactor tool to a **deterministic program transformation calculus**.

---

## Architectural Context

### Where We Are

Under CIPT laws, the system is **stable, correct, and unusually mature**:

- **PHASE 4 Complete**: Visibility & Re-exports enforcement in place
- **PHASE 5 In Progress**: Batch automation with proven N=25 baseline
- **MoveToLayer Frozen**: Action semantics locked and validated
- **Executor Surface Stable**: All enforcement gates operational

### The Gap This Addresses

Current state: **Action-level admission**
- "Is MoveToLayer(X) admissible?" → Yes/No

Required state: **Batch-level admission**
- "Is ⟨A₁, A₂, …, Aₙ⟩ admissible as a sequence?" → Proof or first failure

**Without this schema, Phase 7 learning would violate CIPT principles** by using execution for discovery rather than confirmation.

---

## Core Principles

The schema is governed by four immutable laws:

### 1. Declarative (No Inference)
- All effects must be explicitly declared
- No computed fields, no statistical models, no ML inference
- Unknown effects = inadmissible action by definition

### 2. Conservative (Fail-Safe)
- Over-declaration is safe; under-declaration breaks composition
- If an action *might* touch a surface, it must declare it
- Ambiguity always resolves to "blocked"

### 3. Guard-Equivalent (Runtime Parity)
- Effect declarations must match executor enforcement exactly
- If executor checks a surface, signature must declare it
- No effect can be weaker than runtime validation

### 4. Future-Proof (Versioned Extension)
- New actions add fields via schema versioning
- Existing fields never reinterpreted, weakened, or deprecated
- Schema version 1.0.0 becomes architecturally frozen

---

## Schema Components

### Top-Level Structure

```rust
pub struct EffectSignature {
    pub schema_version: String,        // "0.1.0"
    pub action_type: String,           // "MoveToLayer", "AdjustVisibility"
    pub action_id: String,             // Unique instance identifier
    pub reads: ReadEffects,
    pub writes: WriteEffects,
    pub structural_transitions: StructuralTransitions,
    pub invariant_touchpoints: InvariantTouchpoints,
    pub executor_surface: ExecutorSurface,
}
```

**All fields are mandatory.** Empty collections mean "this action has no reads/writes" (explicit) not "we don't know" (implicit).

---

### 1. Read Effects

Surfaces the action inspects during planning/validation:

```rust
pub struct ReadEffects {
    pub paths: BTreeSet<PathBuf>,                    // Files read
    pub symbols: BTreeSet<String>,                   // Symbols referenced (module::item)
    pub visibility_scopes: BTreeSet<String>,         // Module paths inspected
    pub module_boundaries: BTreeSet<ModuleBoundary>, // Cross-module references
}
```

**Conservatism Rule**: If an action *might* read a surface, declare it. Over-declaration is safe.

---

### 2. Write Effects

Surfaces the action mutates:

```rust
pub struct WriteEffects {
    pub files: BTreeSet<PathBuf>,                         // Files created/modified
    pub modules: BTreeSet<ModuleWrite>,                   // Module declarations
    pub imports: BTreeSet<ImportWrite>,                   // Import statements
    pub re_exports: BTreeSet<ReExportWrite>,              // Re-export (pub use)
    pub visibility_modifiers: BTreeSet<VisibilityWrite>,  // Visibility changes
}
```

**Guard Equivalence**: Write declarations must match executor checks exactly.

---

### 3. Structural Transitions

Architectural shape changes that affect invariant validation:

```rust
pub struct StructuralTransitions {
    pub file_to_module: Vec<FileToModuleTransition>,        // File → module promotions
    pub module_to_layer: Vec<ModuleToLayerTransition>,      // Module → layer movements
    pub test_boundary_crossings: Vec<TestBoundaryCrossing>, // cfg(test) transitions
}
```

These represent phase transitions in codebase structure.

---

### 4. Invariant Touchpoints

Which invariants this action validates against:

```rust
pub struct InvariantTouchpoints {
    pub i1_module_coherence: bool,        // I1: mod declarations match structure
    pub i2_dependency_direction: bool,    // I2: no reverse/circular deps
    pub visibility_law: bool,             // pub exposure matches usage
    pub re_export_law: bool,              // re-exports maintain boundaries
    pub test_topology_law: bool,          // cfg(test) placement rules
}
```

**Exhaustiveness**: Declare ALL invariants the action could violate. Batch composition uses this to detect conflicts.

---

### 5. Executor Surface

Infrastructure gates required for safe execution:

```rust
pub struct ExecutorSurface {
    pub requires_import_repair: bool,             // Auto-add missing imports
    pub requires_module_shim: bool,               // Auto-add module declarations
    pub requires_re_export_enforcement: bool,     // Maintain re-export coherence
    pub requires_verification_gate: bool,         // cargo check after apply
}
```

**Enforcement Contract**:
- If action declares requirement → executor MUST provide it
- If executor provides gate → signature MUST declare it (if used)

---

## Composition Semantics

### Conflict Detection

Two actions conflict if they write to overlapping surfaces in non-commutative ways:

```rust
impl EffectSignature {
    pub fn conflicts_with(&self, other: &EffectSignature) -> bool {
        // File write conflicts
        if !self.writes.files.is_disjoint(&other.writes.files) {
            return true;
        }

        // Module write conflicts
        // Invariant touchpoint conflicts
        // Conservative: same invariant touched = potential conflict
    }
}
```

**Conservative Default**: Touching the same invariant = potential conflict unless proven commutative.

---

### Batch Composition (Future)

Once this schema is locked, batch admission becomes:

```
Σ₀ = initial_state_signature
For each action Aᵢ in batch:
    Σᵢ₊₁ = Σᵢ ⊕ effect(Aᵢ)
    If composition fails:
        Return (inadmissible, i, violated_invariant, chain)

Return (admissible, proof_object)
```

**Abort-First**: Composition fails immediately on first conflict. No reordering, no retries, no speculation.

---

## Schema Versioning

### Current Status: 0.1.0 (Development)

- Pre-freeze development version
- Breaking changes allowed during PHASE 6.5 implementation
- Lock to 1.0.0 when first action declares signatures in production

### Path to 1.0.0

**Freeze Criteria**:
1. MoveToLayer fully declares signatures
2. AdjustVisibility fully declares signatures
3. ReExport fully declares signatures
4. Batch composition validator implemented and tested
5. admission_composition.json artifact proven stable

Once 1.0.0 is released: **Schema is architecturally frozen**.

### Post-1.0.0 Evolution

- New actions add fields: ALLOWED (extend schema to 1.1.0, 1.2.0, etc.)
- Reinterpret existing fields: FORBIDDEN
- Weaken existing semantics: FORBIDDEN
- Add inference/defaults: FORBIDDEN

---

## Usage Constraints

### What You MUST Do

1. **Declare all effects explicitly** - No partial signatures
2. **Use current schema version** - Check `SCHEMA_VERSION` constant
3. **Validate signatures** - Call `.validate()` before admission
4. **Over-declare conservatively** - When in doubt, declare the effect
5. **Match executor exactly** - Guard equivalence is mandatory

### What You MUST NOT Do

1. ❌ **Derive effects by static analysis** - Signatures are hand-written declarations
2. ❌ **Infer effects from prior runs** - No learning from execution
3. ❌ **Allow partial declarations** - All fields mandatory
4. ❌ **Add confidence/probability** - This is deterministic algebra, not statistics
5. ❌ **Execute inadmissible actions** - Admission gate is absolute

---

## Integration Points

### Current System

```
[Correction Intelligence] → [Action Candidates]
                               ↓
[Admission Preflight] → admissible: true/false
                               ↓
[Executor] (only if admissible)
```

### PHASE 6.5 Target

```
[Correction Intelligence] → [Action Candidates + Effect Signatures]
                                      ↓
[Batch Composition Validator] → admission_composition.json
                                      ↓
[Admission Preflight] → proof object or first_failure
                                      ↓
[Executor] (only if batch admissible)
```

---

## Explicit Prohibitions

Do **NOT** proceed with the following until this schema is locked and batch composition is operational:

- ❌ Error-pattern clustering from execution logs
- ❌ Automatic allowlist expansion based on success rates
- ❌ Constraint suggestion from failure messages
- ❌ Probabilistic admission scoring
- ❌ ML-based effect inference

**These become derivative optimizations after formal foundation exists.**

---

## Success Criteria

PHASE 6.5 is complete when:

1. ✅ Static Effect Signature Schema locked at 1.0.0
2. ✅ All active actions declare complete signatures
3. ✅ Batch composition validator implemented
4. ✅ admission_composition.json artifact generated on every run
5. ✅ Proof objects trusted for human audit
6. ✅ No execution used for learning (confirmation only)

---

## Philosophical Statement

> You are no longer building a refactor tool.
> You are building a **deterministic program transformation calculus**.
> The next correct move is not more automation —
> it is **formal compositional truth at admission time**.

---

## References

- **CIPT Process**: `Claude.md` (Operating Principles)
- **Phase Tracker**: `Cluster_progress.txt` (PHASE 6.5 entry)
- **Schema Implementation**: `src/effect_signature_schema.rs`
- **Constraint Harvest**: `docs/97_correction_intelligence/constraint_harvest_log.md`

---

**END OF SPECIFICATION**
