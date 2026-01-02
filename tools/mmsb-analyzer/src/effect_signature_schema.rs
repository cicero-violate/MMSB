//! Static Effect Signature Schema (PHASE 6.5 Foundation)
//!
//! # ARCHITECTURAL FREEZE
//!
//! PHASE 6.5 Admission Intelligence is complete.
//! No changes permitted without explicit architectural authorization.
//!
//! # CIPT Compliance
//!
//! This schema is the **keystone** of PHASE 6.5: Admission Intelligence Formalization.
//! It defines the formal language of transformation effects, enabling compositional
//! reasoning about action sequences without execution.
//!
//! # Schema Status: FROZEN
//!
//! Once this schema reaches version 1.0.0, it is **architecturally frozen**.
//! New actions may add fields via schema versioning, but existing fields must never
//! be reinterpreted, weakened, or made inferential.
//!
//! # Core Principles
//!
//! 1. **Declarative**: No computed fields, no inference, no defaults
//! 2. **Conservative**: Unknown effects = inadmissible action
//! 3. **Guard-Equivalent**: Never weaker than executor runtime checks
//! 4. **Future-Proof**: Extensible via versioning, never via reinterpretation
//!
//! # Invariant
//!
//! If an action cannot fully declare its effect signature, it is **inadmissible by definition**.
//!
//! # Version History
//!
//! - 0.1.0: Initial schema definition (pre-freeze, development only)

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Schema version marker
pub const SCHEMA_VERSION: &str = "0.1.0";

/// Top-level effect signature for a transformation action
///
/// # Completeness Requirement
///
/// All fields are **mandatory**. Empty sets are explicit ("this action has no reads")
/// rather than implicit ("we don't know what this action reads").
///
/// An action that cannot populate all fields is inadmissible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSignature {
    /// Schema version this signature conforms to
    pub schema_version: String,

    /// Action identifier (e.g., "MoveToLayer", "AdjustVisibility")
    pub action_type: String,

    /// Unique identifier for this specific action instance
    pub action_id: String,

    /// What this action reads (must be exhaustive)
    pub reads: ReadEffects,

    /// What this action writes (must be exhaustive)
    pub writes: WriteEffects,

    /// Structural transitions this action performs
    pub structural_transitions: StructuralTransitions,

    /// Invariants this action touches or validates against
    pub invariant_touchpoints: InvariantTouchpoints,

    /// Executor infrastructure this action requires
    pub executor_surface: ExecutorSurface,
}

/// Read effects: surfaces the action inspects
///
/// # Conservatism
///
/// If an action *might* read a surface, it must declare it.
/// Over-declaration is safe; under-declaration breaks composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadEffects {
    /// File paths read (absolute paths from project root)
    pub paths: BTreeSet<PathBuf>,

    /// Symbols referenced (fully qualified: module::item)
    pub symbols: BTreeSet<String>,

    /// Visibility scopes inspected (module paths)
    pub visibility_scopes: BTreeSet<String>,

    /// Module boundaries traversed during analysis
    pub module_boundaries: BTreeSet<ModuleBoundary>,
}

/// Write effects: surfaces the action mutates
///
/// # Guard Equivalence
///
/// Write declarations must match executor enforcement exactly.
/// If executor checks a surface, the signature must declare it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteEffects {
    /// Files created or modified
    pub files: BTreeSet<PathBuf>,

    /// Modules created or modified
    pub modules: BTreeSet<ModuleWrite>,

    /// Imports added or modified
    pub imports: BTreeSet<ImportWrite>,

    /// Re-exports added or modified
    pub re_exports: BTreeSet<ReExportWrite>,

    /// Visibility modifiers changed
    pub visibility_modifiers: BTreeSet<VisibilityWrite>,
}

/// Structural transitions: architectural shape changes
///
/// These represent phase transitions in the codebase structure that
/// affect invariant validation and dependency reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralTransitions {
    /// File → module promotions
    pub file_to_module: Vec<FileToModuleTransition>,

    /// Module → layer movements
    pub module_to_layer: Vec<ModuleToLayerTransition>,

    /// Test boundary crossings (#[cfg(test)] or #[test] transitions)
    pub test_boundary_crossings: Vec<TestBoundaryCrossing>,
}

/// Invariant touchpoints: which invariants this action validates against
///
/// # Exhaustiveness
///
/// An action must declare ALL invariants it could violate.
/// Batch composition uses this to detect invariant conflicts between actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantTouchpoints {
    /// I1: Module coherence (mod declarations match file structure)
    pub i1_module_coherence: bool,

    /// I2: Dependency direction (no reverse or circular dependencies)
    pub i2_dependency_direction: bool,

    /// Visibility law (pub exposure matches usage)
    pub visibility_law: bool,

    /// Re-export law (re-exports maintain module boundaries)
    pub re_export_law: bool,

    /// Test topology law (cfg(test) placement rules)
    pub test_topology_law: bool,
}

/// Executor surface: infrastructure gates required for safe execution
///
/// # Enforcement Contract
///
/// If an action declares a surface requirement, executor MUST provide it.
/// If executor provides a gate, signature MUST declare it if action uses it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorSurface {
    /// Requires import repair (auto-add missing imports)
    pub requires_import_repair: bool,

    /// Requires module shim (auto-add module declarations)
    pub requires_module_shim: bool,

    /// Requires re-export enforcement (maintain re-export coherence)
    pub requires_re_export_enforcement: bool,

    /// Requires verification gate (cargo check after apply)
    pub requires_verification_gate: bool,
}

// === Supporting Types ===

/// Module boundary crossing descriptor
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModuleBoundary {
    /// Source module path
    pub from_module: String,

    /// Target module path
    pub to_module: String,

    /// Type of boundary crossing
    pub crossing_type: BoundaryCrossingType,
}

/// Type of module boundary crossing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BoundaryCrossingType {
    /// Normal import/use
    Import,

    /// Re-export (pub use)
    ReExport,

    /// Symbol visibility extension
    VisibilityExtension,
}

/// Module write operation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ModuleWrite {
    /// Module path being written
    pub module_path: String,

    /// File where module declaration appears
    pub declaration_file: PathBuf,

    /// Operation type
    pub operation: ModuleOperation,
}

/// Module operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModuleOperation {
    /// Create new module declaration
    Create,

    /// Modify existing module declaration
    Modify,

    /// Remove module declaration
    Remove,
}

/// Import write operation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ImportWrite {
    /// Target file receiving the import
    pub target_file: PathBuf,

    /// Import path (e.g., "crate::module::item")
    pub import_path: String,

    /// Whether this is a re-export (pub use)
    pub is_re_export: bool,
}

/// Re-export write operation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReExportWrite {
    /// File containing the re-export
    pub file: PathBuf,

    /// Symbol being re-exported
    pub symbol: String,

    /// Source module of the symbol
    pub from_module: String,

    /// Target visibility (pub, pub(crate), etc.)
    pub visibility: String,
}

/// Visibility modifier write
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VisibilityWrite {
    /// File containing the symbol
    pub file: PathBuf,

    /// Symbol whose visibility changes
    pub symbol: String,

    /// Old visibility modifier
    pub old_visibility: String,

    /// New visibility modifier
    pub new_visibility: String,
}

/// File to module transition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileToModuleTransition {
    /// Original file path
    pub file_path: PathBuf,

    /// New module path
    pub module_path: String,
}

/// Module to layer transition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleToLayerTransition {
    /// Module being moved
    pub module_path: String,

    /// Source layer (numeric or named)
    pub from_layer: String,

    /// Target layer (numeric or named)
    pub to_layer: String,
}

/// Test boundary crossing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestBoundaryCrossing {
    /// Symbol crossing the boundary
    pub symbol: String,

    /// Direction of crossing
    pub direction: TestBoundaryDirection,

    /// File where crossing occurs
    pub file: PathBuf,
}

/// Test boundary crossing direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestBoundaryDirection {
    /// Non-test → test
    IntoTest,

    /// Test → non-test
    OutOfTest,
}

// === Validation ===

impl EffectSignature {
    /// Validate that this signature is complete and well-formed
    ///
    /// # Returns
    ///
    /// - `Ok(())` if signature is valid
    /// - `Err(reason)` if signature is incomplete or malformed
    pub fn validate(&self) -> Result<(), String> {
        // Schema version must match current version
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "Schema version mismatch: expected {}, got {}",
                SCHEMA_VERSION, self.schema_version
            ));
        }

        // Action type must be non-empty
        if self.action_type.is_empty() {
            return Err("action_type cannot be empty".to_string());
        }

        // Action ID must be non-empty
        if self.action_id.is_empty() {
            return Err("action_id cannot be empty".to_string());
        }

        // No partial declarations allowed - all fields must be explicitly set
        // (This is enforced by type system - all fields are mandatory)

        Ok(())
    }

    /// Check if this signature conflicts with another on write surfaces
    ///
    /// Two actions conflict if they write to overlapping surfaces in ways
    /// that cannot be proven commutative.
    pub fn conflicts_with(&self, other: &EffectSignature) -> bool {
        // File write conflicts
        if !self.writes.files.is_disjoint(&other.writes.files) {
            return true;
        }

        // Module write conflicts
        let self_modules: BTreeSet<_> = self
            .writes
            .modules
            .iter()
            .map(|m| &m.module_path)
            .collect();
        let other_modules: BTreeSet<_> = other
            .writes
            .modules
            .iter()
            .map(|m| &m.module_path)
            .collect();
        if !self_modules.is_disjoint(&other_modules) {
            return true;
        }

        // Invariant touchpoint conflicts
        // If both touch the same invariant, they potentially conflict
        let self_inv = &self.invariant_touchpoints;
        let other_inv = &other.invariant_touchpoints;

        if (self_inv.i1_module_coherence && other_inv.i1_module_coherence)
            || (self_inv.i2_dependency_direction && other_inv.i2_dependency_direction)
            || (self_inv.visibility_law && other_inv.visibility_law)
            || (self_inv.re_export_law && other_inv.re_export_law)
            || (self_inv.test_topology_law && other_inv.test_topology_law)
        {
            // Conservative: touching same invariant = potential conflict
            // Future refinement can prove commutativity for specific cases
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_validation() {
        let sig = EffectSignature {
            schema_version: SCHEMA_VERSION.to_string(),
            action_type: "MoveToLayer".to_string(),
            action_id: "move_001".to_string(),
            reads: ReadEffects {
                paths: BTreeSet::new(),
                symbols: BTreeSet::new(),
                visibility_scopes: BTreeSet::new(),
                module_boundaries: BTreeSet::new(),
            },
            writes: WriteEffects {
                files: BTreeSet::new(),
                modules: BTreeSet::new(),
                imports: BTreeSet::new(),
                re_exports: BTreeSet::new(),
                visibility_modifiers: BTreeSet::new(),
            },
            structural_transitions: StructuralTransitions {
                file_to_module: Vec::new(),
                module_to_layer: Vec::new(),
                test_boundary_crossings: Vec::new(),
            },
            invariant_touchpoints: InvariantTouchpoints {
                i1_module_coherence: true,
                i2_dependency_direction: true,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
            executor_surface: ExecutorSurface {
                requires_import_repair: true,
                requires_module_shim: true,
                requires_re_export_enforcement: false,
                requires_verification_gate: true,
            },
        };

        assert!(sig.validate().is_ok());
    }

    #[test]
    fn test_signature_conflict_detection() {
        let sig1 = create_test_signature("action1", "file1.rs");
        let sig2 = create_test_signature("action2", "file2.rs");
        let sig3 = create_test_signature("action3", "file1.rs"); // Same file as sig1

        assert!(!sig1.conflicts_with(&sig2), "Different files should not conflict");
        assert!(sig1.conflicts_with(&sig3), "Same file should conflict");
    }

    fn create_test_signature(action_id: &str, file: &str) -> EffectSignature {
        let mut files = BTreeSet::new();
        files.insert(PathBuf::from(file));

        EffectSignature {
            schema_version: SCHEMA_VERSION.to_string(),
            action_type: "Test".to_string(),
            action_id: action_id.to_string(),
            reads: ReadEffects {
                paths: BTreeSet::new(),
                symbols: BTreeSet::new(),
                visibility_scopes: BTreeSet::new(),
                module_boundaries: BTreeSet::new(),
            },
            writes: WriteEffects {
                files,
                modules: BTreeSet::new(),
                imports: BTreeSet::new(),
                re_exports: BTreeSet::new(),
                visibility_modifiers: BTreeSet::new(),
            },
            structural_transitions: StructuralTransitions {
                file_to_module: Vec::new(),
                module_to_layer: Vec::new(),
                test_boundary_crossings: Vec::new(),
            },
            invariant_touchpoints: InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
            executor_surface: ExecutorSurface {
                requires_import_repair: false,
                requires_module_shim: false,
                requires_re_export_enforcement: false,
                requires_verification_gate: false,
            },
        }
    }
}
