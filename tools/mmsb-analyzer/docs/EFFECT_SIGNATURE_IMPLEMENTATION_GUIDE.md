# Effect Signature Implementation Guide

**Target Audience**: Developers implementing transformation actions
**Status**: CIPT Foundation - PHASE 6.5
**Version**: 0.1.0

---

## Purpose

This guide shows how to declare Static Effect Signatures for transformation actions in compliance with CIPT principles. Every action must declare its effects explicitly before it can participate in batch composition.

---

## Quick Start

### 1. Import the Schema

```rust
use mmsb_analyzer::effect_signature_schema::{
    EffectSignature, ReadEffects, WriteEffects,
    StructuralTransitions, InvariantTouchpoints,
    ExecutorSurface, SCHEMA_VERSION,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
```

### 2. Declare Your Action's Effects

```rust
impl MyAction {
    pub fn effect_signature(&self) -> EffectSignature {
        EffectSignature {
            schema_version: SCHEMA_VERSION.to_string(),
            action_type: "MyAction".to_string(),
            action_id: format!("myaction_{}", self.id),
            reads: self.declare_reads(),
            writes: self.declare_writes(),
            structural_transitions: self.declare_transitions(),
            invariant_touchpoints: self.declare_invariants(),
            executor_surface: self.declare_executor_needs(),
        }
    }
}
```

### 3. Validate Before Use

```rust
let signature = action.effect_signature();
signature.validate()
    .expect("Effect signature must be complete and valid");
```

---

## Implementation Checklist

For each transformation action, you must:

- [ ] Declare all file reads (paths)
- [ ] Declare all symbol references
- [ ] Declare all visibility scopes inspected
- [ ] Declare all module boundaries crossed
- [ ] Declare all file writes
- [ ] Declare all module writes
- [ ] Declare all import writes
- [ ] Declare all re-export writes
- [ ] Declare all visibility modifier changes
- [ ] Declare all structural transitions
- [ ] Declare all invariant touchpoints
- [ ] Declare all executor surface requirements
- [ ] Validate signature completeness
- [ ] Test signature conflict detection

**No partial declarations allowed.** Empty sets are explicit, not omissions.

---

## Examples by Action Type

### Example 1: MoveToLayer

```rust
impl MoveToLayer {
    fn declare_reads(&self) -> ReadEffects {
        let mut paths = BTreeSet::new();
        let mut symbols = BTreeSet::new();
        let mut visibility_scopes = BTreeSet::new();
        let mut module_boundaries = BTreeSet::new();

        // Read the source file
        paths.insert(self.source_file.clone());

        // Read symbols being moved
        for symbol in &self.symbols {
            symbols.insert(symbol.clone());
        }

        // Inspect source module visibility
        visibility_scopes.insert(self.source_module.clone());

        // Cross module boundary
        module_boundaries.insert(ModuleBoundary {
            from_module: self.source_module.clone(),
            to_module: self.target_module.clone(),
            crossing_type: BoundaryCrossingType::Import,
        });

        ReadEffects {
            paths,
            symbols,
            visibility_scopes,
            module_boundaries,
        }
    }

    fn declare_writes(&self) -> WriteEffects {
        let mut files = BTreeSet::new();
        let mut modules = BTreeSet::new();
        let mut imports = BTreeSet::new();

        // Write to target layer file
        files.insert(self.target_file.clone());

        // May modify source file (remove code)
        files.insert(self.source_file.clone());

        // Create or modify target module
        modules.insert(ModuleWrite {
            module_path: self.target_module.clone(),
            declaration_file: self.target_layer_mod_file.clone(),
            operation: ModuleOperation::Modify,
        });

        // Add imports to target file
        for import in &self.required_imports {
            imports.insert(ImportWrite {
                target_file: self.target_file.clone(),
                import_path: import.clone(),
                is_re_export: false,
            });
        }

        WriteEffects {
            files,
            modules,
            imports,
            re_exports: BTreeSet::new(),  // Explicit: no re-exports
            visibility_modifiers: BTreeSet::new(),  // Explicit: no visibility changes
        }
    }

    fn declare_transitions(&self) -> StructuralTransitions {
        let module_to_layer = vec![ModuleToLayerTransition {
            module_path: self.symbol_module.clone(),
            from_layer: self.source_layer.clone(),
            to_layer: self.target_layer.clone(),
        }];

        StructuralTransitions {
            file_to_module: Vec::new(),  // Explicit: no file→module promotions
            module_to_layer,
            test_boundary_crossings: Vec::new(),  // Explicit: no test crossings
        }
    }

    fn declare_invariants(&self) -> InvariantTouchpoints {
        InvariantTouchpoints {
            i1_module_coherence: true,      // Touches module structure
            i2_dependency_direction: true,  // Affects dependency graph
            visibility_law: false,          // Doesn't change visibility
            re_export_law: false,           // Doesn't touch re-exports
            test_topology_law: false,       // Doesn't cross test boundaries
        }
    }

    fn declare_executor_needs(&self) -> ExecutorSurface {
        ExecutorSurface {
            requires_import_repair: true,            // Needs auto-import
            requires_module_shim: true,              // Needs module declarations
            requires_re_export_enforcement: false,   // No re-export work
            requires_verification_gate: true,        // Must verify after apply
        }
    }
}
```

---

### Example 2: AdjustVisibility

```rust
impl AdjustVisibility {
    fn declare_reads(&self) -> ReadEffects {
        let mut paths = BTreeSet::new();
        let mut symbols = BTreeSet::new();
        let mut visibility_scopes = BTreeSet::new();

        // Read the file containing the symbol
        paths.insert(self.file.clone());

        // Read the symbol being modified
        symbols.insert(self.symbol.clone());

        // Inspect current visibility scope
        visibility_scopes.insert(self.current_module.clone());

        ReadEffects {
            paths,
            symbols,
            visibility_scopes,
            module_boundaries: BTreeSet::new(),  // No boundary crossings
        }
    }

    fn declare_writes(&self) -> WriteEffects {
        let mut files = BTreeSet::new();
        let mut visibility_modifiers = BTreeSet::new();

        // Modify the file
        files.insert(self.file.clone());

        // Change visibility
        visibility_modifiers.insert(VisibilityWrite {
            file: self.file.clone(),
            symbol: self.symbol.clone(),
            old_visibility: self.old_visibility.clone(),
            new_visibility: self.new_visibility.clone(),
        });

        WriteEffects {
            files,
            modules: BTreeSet::new(),
            imports: BTreeSet::new(),
            re_exports: BTreeSet::new(),
            visibility_modifiers,
        }
    }

    fn declare_transitions(&self) -> StructuralTransitions {
        StructuralTransitions {
            file_to_module: Vec::new(),
            module_to_layer: Vec::new(),
            test_boundary_crossings: Vec::new(),
        }
    }

    fn declare_invariants(&self) -> InvariantTouchpoints {
        InvariantTouchpoints {
            i1_module_coherence: false,
            i2_dependency_direction: false,
            visibility_law: true,          // Primary invariant
            re_export_law: false,
            test_topology_law: false,
        }
    }

    fn declare_executor_needs(&self) -> ExecutorSurface {
        ExecutorSurface {
            requires_import_repair: false,
            requires_module_shim: false,
            requires_re_export_enforcement: false,
            requires_verification_gate: true,
        }
    }
}
```

---

## CIPT Compliance Rules

### Rule 1: No Inference

```rust
// ❌ WRONG: Computed/inferred effects
fn declare_reads(&self) -> ReadEffects {
    let paths = self.analyze_dependencies();  // NO!
    ReadEffects { paths, .. }
}

// ✅ CORRECT: Explicit declaration
fn declare_reads(&self) -> ReadEffects {
    let mut paths = BTreeSet::new();
    paths.insert(self.known_source_file.clone());
    ReadEffects { paths, .. }
}
```

### Rule 2: Conservative Over-Declaration

```rust
// When in doubt, declare it
fn declare_reads(&self) -> ReadEffects {
    let mut paths = BTreeSet::new();

    // Even if we might not read it, declare it
    if self.may_need_config_file {
        paths.insert(PathBuf::from("config.toml"));
    }

    ReadEffects { paths, .. }
}
```

### Rule 3: Guard Equivalence

```rust
// If executor checks it, signature must declare it
impl MyAction {
    fn declare_executor_needs(&self) -> ExecutorSurface {
        ExecutorSurface {
            // If executor runs cargo check after this action,
            // you MUST declare it
            requires_verification_gate: true,
            ..
        }
    }
}
```

### Rule 4: Explicit Empties

```rust
// ✅ CORRECT: Explicit empty set
WriteEffects {
    files: BTreeSet::new(),  // "This action writes no files"
    modules: BTreeSet::new(), // "This action writes no modules"
    ..
}

// ❌ WRONG: Field omission (won't compile anyway)
WriteEffects {
    files: my_files,
    // Missing other fields - compiler error
}
```

---

## Testing Your Signature

### Test 1: Validation

```rust
#[test]
fn test_signature_validation() {
    let action = MyAction::new(..);
    let sig = action.effect_signature();

    assert!(sig.validate().is_ok(), "Signature must be valid");
    assert_eq!(sig.schema_version, SCHEMA_VERSION);
    assert!(!sig.action_type.is_empty());
    assert!(!sig.action_id.is_empty());
}
```

### Test 2: Conflict Detection

```rust
#[test]
fn test_signature_conflicts() {
    let action1 = MyAction::new_for_file("src/foo.rs");
    let action2 = MyAction::new_for_file("src/bar.rs");
    let action3 = MyAction::new_for_file("src/foo.rs");

    let sig1 = action1.effect_signature();
    let sig2 = action2.effect_signature();
    let sig3 = action3.effect_signature();

    assert!(!sig1.conflicts_with(&sig2), "Different files should not conflict");
    assert!(sig1.conflicts_with(&sig3), "Same file should conflict");
}
```

### Test 3: Completeness

```rust
#[test]
fn test_signature_completeness() {
    let action = MyAction::new(..);
    let sig = action.effect_signature();

    // Verify all expected effects are declared
    assert!(!sig.writes.files.is_empty(), "Must declare file writes");
    assert!(sig.invariant_touchpoints.i1_module_coherence,
            "Must touch I1 invariant");
    assert!(sig.executor_surface.requires_verification_gate,
            "Must require verification");
}
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Secondary Reads

```rust
// ❌ Missing dependency file reads
fn declare_reads(&self) -> ReadEffects {
    let mut paths = BTreeSet::new();
    paths.insert(self.primary_file.clone());
    // FORGOT: We also read imported modules!
    ReadEffects { paths, .. }
}

// ✅ Complete declaration
fn declare_reads(&self) -> ReadEffects {
    let mut paths = BTreeSet::new();
    paths.insert(self.primary_file.clone());
    // Include all transitive reads
    for dep in &self.dependencies {
        paths.insert(dep.file.clone());
    }
    ReadEffects { paths, .. }
}
```

### Pitfall 2: Under-Declaring Invariants

```rust
// ❌ Missing invariant touchpoint
InvariantTouchpoints {
    i1_module_coherence: true,
    // FORGOT: Moving layers also affects dependency direction!
    i2_dependency_direction: false,  // Should be true
    ..
}

// ✅ Complete invariant declaration
InvariantTouchpoints {
    i1_module_coherence: true,
    i2_dependency_direction: true,  // Affects dep graph
    ..
}
```

### Pitfall 3: Optimistic File Writes

```rust
// ❌ Assuming no writes to source
WriteEffects {
    files: vec![self.target_file.clone()].into_iter().collect(),
    // FORGOT: We might modify source file too (remove moved code)
    ..
}

// ✅ Conservative declaration
WriteEffects {
    files: vec![
        self.target_file.clone(),
        self.source_file.clone(),  // May be modified
    ].into_iter().collect(),
    ..
}
```

---

## Integration Checklist

Before your action participates in batch composition:

1. [ ] Effect signature fully implemented
2. [ ] All reads explicitly declared
3. [ ] All writes explicitly declared
4. [ ] All structural transitions declared
5. [ ] All invariant touchpoints declared
6. [ ] All executor requirements declared
7. [ ] Signature validation passes
8. [ ] Conflict detection tested
9. [ ] Completeness verified
10. [ ] Documentation updated

---

## FAQ

**Q: What if I don't know all the files my action will read?**
A: Then your action is inadmissible. Effects must be knowable at declaration time.

**Q: Can I use static analysis to compute effects?**
A: No. Effects are explicit declarations, not inferred properties.

**Q: What if my action might or might not write a file?**
A: Declare it conservatively. Over-declaration is safe; under-declaration breaks composition.

**Q: Can I declare partial effects and refine later?**
A: No. All fields are mandatory. Unknown effects = inadmissible action.

**Q: How do I handle dynamic file paths?**
A: Actions with dynamic paths are inadmissible until paths are concrete and knowable.

**Q: Can I skip empty collections?**
A: No. Empty collections must be explicitly provided (they mean "no effects of this type").

---

## References

- **Schema Specification**: `docs/PHASE_6_5_EFFECT_SIGNATURE_SCHEMA.md`
- **JSON Schema**: `docs/effect_signature_schema.json`
- **Rust Implementation**: `src/effect_signature_schema.rs`
- **CIPT Principles**: `Claude.md`
- **Phase Tracker**: `Cluster_progress.txt`

---

**Remember**: You are not building heuristics. You are building deterministic proof objects.

**END OF GUIDE**
