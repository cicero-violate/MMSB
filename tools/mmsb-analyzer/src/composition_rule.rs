//! Composition Rule (Abort-First) - PHASE 6.5 Activation
//!
//! # ARCHITECTURAL FREEZE
//!
//! PHASE 6.5 Admission Intelligence is complete.
//! No changes permitted without explicit architectural authorization.
//!
//! # Purpose
//!
//! Implements the deterministic fold that enables batch-level admission.
//! This is the activation step of PHASE 6.5: Admission Intelligence Formalization.
//!
//! # Guarantees
//!
//! - **Pure**: No side effects, read-only computation
//! - **Abort-First**: Fails immediately on first conflict
//! - **Order-Sensitive**: Sequence order matters (no reordering)
//! - **Non-Speculative**: No retries, no backtracking, no "maybe"
//! - **Conservative**: Unknown commutativity = conflict
//!
//! # Question Answered
//!
//! Given a concrete ordered sequence ⟨A₁…Aₙ⟩, is the sequence admissible
//! under frozen invariants?
//!
//! # Version
//!
//! - 0.1.0: Initial implementation (pre-freeze)

use crate::effect_signature_schema::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

/// Result of batch composition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionResult {
    /// Batch is admissible
    Admissible {
        /// Number of actions in the batch
        action_count: usize,
        /// Final composed state (for audit)
        final_state: ComposedEffectState,
    },
    /// Batch is inadmissible (aborted at first conflict)
    Inadmissible {
        /// Index of the action that caused the conflict (0-based)
        first_failure_index: usize,
        /// The action that failed
        failed_action_id: String,
        /// Why it failed
        conflict_reason: ConflictReason,
        /// State before the failed action
        state_before_failure: ComposedEffectState,
    },
}

/// Reason for composition conflict
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictReason {
    /// Two actions write to the same file
    FileWriteConflict {
        file: PathBuf,
        prior_action_index: usize,
    },
    /// Two actions write to the same module
    ModuleWriteConflict {
        module_path: String,
        prior_action_index: usize,
    },
    /// Action reads a surface written by a prior action
    ReadAfterWriteAmbiguity {
        surface: String,
        written_by_index: usize,
    },
    /// Two actions touch the same invariant
    InvariantOverlap {
        invariant: InvariantType,
        prior_action_index: usize,
    },
    /// Executor surface forbidden
    ExecutorSurfaceForbidden {
        surface: String,
    },
}

/// Invariant types for conflict reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InvariantType {
    I1ModuleCoherence,
    I2DependencyDirection,
    VisibilityLaw,
    ReExportLaw,
    TestTopologyLaw,
}

/// Symbolic shadow state of accumulated effects
///
/// This is **not** the program state. It is a symbolic representation
/// of what effects have been accumulated during composition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposedEffectState {
    /// Files written (maps file to action index that wrote it)
    pub files_written: BTreeMap<PathBuf, usize>,

    /// Modules written (maps module path to action index)
    pub modules_written: BTreeMap<String, usize>,

    /// Imports written (maps (target_file, import_path) to action index)
    pub imports_written: BTreeMap<(PathBuf, String), usize>,

    /// Re-exports written (maps (file, symbol) to action index)
    pub re_exports_written: BTreeMap<(PathBuf, String), usize>,

    /// Visibility modifiers written (maps (file, symbol) to action index)
    pub visibility_modifiers_written: BTreeMap<(PathBuf, String), usize>,

    /// Files read (for read-after-write detection)
    pub files_read: BTreeSet<PathBuf>,

    /// Symbols read (for read-after-write detection)
    pub symbols_read: BTreeSet<String>,

    /// Invariants touched (maps invariant to action index that touched it)
    pub invariants_touched: BTreeMap<InvariantType, Vec<usize>>,

    /// Executor surfaces required
    pub executor_surfaces: ExecutorSurface,

    /// Number of actions composed so far
    pub action_count: usize,
}

impl ComposedEffectState {
    /// Create empty initial state
    pub fn empty() -> Self {
        Self {
            files_written: BTreeMap::new(),
            modules_written: BTreeMap::new(),
            imports_written: BTreeMap::new(),
            re_exports_written: BTreeMap::new(),
            visibility_modifiers_written: BTreeMap::new(),
            files_read: BTreeSet::new(),
            symbols_read: BTreeSet::new(),
            invariants_touched: BTreeMap::new(),
            executor_surfaces: ExecutorSurface {
                requires_import_repair: false,
                requires_module_shim: false,
                requires_re_export_enforcement: false,
                requires_verification_gate: false,
            },
            action_count: 0,
        }
    }
}

/// Compose a batch of actions in order, aborting on first conflict
///
/// # Arguments
///
/// * `batch` - Ordered sequence of effect signatures
///
/// # Returns
///
/// * `CompositionResult::Admissible` if entire batch is conflict-free
/// * `CompositionResult::Inadmissible` with first conflict details
///
/// # Guarantees
///
/// - Pure function (no side effects)
/// - Deterministic (same input → same output)
/// - Abort-first (stops at first conflict)
/// - Order-sensitive (reordering may change result)
/// - Conservative (unknown commutativity → conflict)
pub fn compose_batch(batch: &[EffectSignature]) -> CompositionResult {
    let mut state = ComposedEffectState::empty();

    for (index, signature) in batch.iter().enumerate() {
        // Check for conflicts with accumulated state
        if let Some(conflict) = check_conflicts(&state, signature, index) {
            return CompositionResult::Inadmissible {
                first_failure_index: index,
                failed_action_id: signature.action_id.clone(),
                conflict_reason: conflict,
                state_before_failure: state,
            };
        }

        // No conflict - compose this action into the state
        compose_into_state(&mut state, signature, index);
    }

    // All actions composed successfully
    CompositionResult::Admissible {
        action_count: batch.len(),
        final_state: state,
    }
}

/// Check if a signature conflicts with accumulated state
///
/// Returns `Some(ConflictReason)` on first conflict, `None` if compatible.
///
/// # Conservative Rules
///
/// 1. Write/Write overlap → conflict
/// 2. Read-after-write → conflict (no commutativity proof yet)
/// 3. Invariant overlap → conflict (no commutativity proof yet)
/// 4. Executor surface escalation → allowed (monotonic)
fn check_conflicts(
    state: &ComposedEffectState,
    signature: &EffectSignature,
    _current_index: usize,
) -> Option<ConflictReason> {
    // Rule 1a: File write conflicts
    for file in &signature.writes.files {
        if let Some(&prior_index) = state.files_written.get(file) {
            return Some(ConflictReason::FileWriteConflict {
                file: file.clone(),
                prior_action_index: prior_index,
            });
        }
    }

    // Rule 1b: Module write conflicts
    for module_write in &signature.writes.modules {
        if let Some(&prior_index) = state.modules_written.get(&module_write.module_path) {
            return Some(ConflictReason::ModuleWriteConflict {
                module_path: module_write.module_path.clone(),
                prior_action_index: prior_index,
            });
        }
    }

    // Rule 1c: Import write conflicts
    for import_write in &signature.writes.imports {
        let key = (import_write.target_file.clone(), import_write.import_path.clone());
        if let Some(&prior_index) = state.imports_written.get(&key) {
            return Some(ConflictReason::ReadAfterWriteAmbiguity {
                surface: format!("import {} in {:?}", import_write.import_path, import_write.target_file),
                written_by_index: prior_index,
            });
        }
    }

    // Rule 1d: Re-export write conflicts
    for re_export_write in &signature.writes.re_exports {
        let key = (re_export_write.file.clone(), re_export_write.symbol.clone());
        if let Some(&prior_index) = state.re_exports_written.get(&key) {
            return Some(ConflictReason::ReadAfterWriteAmbiguity {
                surface: format!("re-export {} in {:?}", re_export_write.symbol, re_export_write.file),
                written_by_index: prior_index,
            });
        }
    }

    // Rule 1e: Visibility modifier conflicts
    for visibility_write in &signature.writes.visibility_modifiers {
        let key = (visibility_write.file.clone(), visibility_write.symbol.clone());
        if let Some(&prior_index) = state.visibility_modifiers_written.get(&key) {
            return Some(ConflictReason::ReadAfterWriteAmbiguity {
                surface: format!("visibility {} in {:?}", visibility_write.symbol, visibility_write.file),
                written_by_index: prior_index,
            });
        }
    }

    // Rule 2: Read-after-write ambiguity
    // Conservative: if this action reads a file that was written by any prior action, conflict
    for read_path in &signature.reads.paths {
        if let Some(&prior_index) = state.files_written.get(read_path) {
            return Some(ConflictReason::ReadAfterWriteAmbiguity {
                surface: format!("file {:?}", read_path),
                written_by_index: prior_index,
            });
        }
    }

    // Rule 3: Invariant overlap
    // Conservative: if this action touches an invariant already touched, conflict
    // (Future refinement can prove commutativity for specific cases)

    let current_invariants = collect_invariants_touched(&signature.invariant_touchpoints);

    for invariant in current_invariants {
        if let Some(prior_indices) = state.invariants_touched.get(&invariant) {
            if !prior_indices.is_empty() {
                return Some(ConflictReason::InvariantOverlap {
                    invariant,
                    prior_action_index: prior_indices[0],
                });
            }
        }
    }

    // Rule 4: Executor surface escalation
    // Executor requirements are monotonic - new requirements are allowed
    // Only abort if executor explicitly forbids a surface (not implemented yet)

    None
}

/// Compose a signature into the accumulated state
///
/// This is called only after conflict checking passes.
fn compose_into_state(
    state: &mut ComposedEffectState,
    signature: &EffectSignature,
    action_index: usize,
) {
    // Record file writes
    for file in &signature.writes.files {
        state.files_written.insert(file.clone(), action_index);
    }

    // Record module writes
    for module_write in &signature.writes.modules {
        state.modules_written.insert(module_write.module_path.clone(), action_index);
    }

    // Record import writes
    for import_write in &signature.writes.imports {
        let key = (import_write.target_file.clone(), import_write.import_path.clone());
        state.imports_written.insert(key, action_index);
    }

    // Record re-export writes
    for re_export_write in &signature.writes.re_exports {
        let key = (re_export_write.file.clone(), re_export_write.symbol.clone());
        state.re_exports_written.insert(key, action_index);
    }

    // Record visibility modifier writes
    for visibility_write in &signature.writes.visibility_modifiers {
        let key = (visibility_write.file.clone(), visibility_write.symbol.clone());
        state.visibility_modifiers_written.insert(key, action_index);
    }

    // Record reads
    for path in &signature.reads.paths {
        state.files_read.insert(path.clone());
    }
    for symbol in &signature.reads.symbols {
        state.symbols_read.insert(symbol.clone());
    }

    // Record invariant touchpoints
    let invariants = collect_invariants_touched(&signature.invariant_touchpoints);
    for invariant in invariants {
        state.invariants_touched
            .entry(invariant)
            .or_insert_with(Vec::new)
            .push(action_index);
    }

    // Accumulate executor surface requirements (monotonic)
    state.executor_surfaces.requires_import_repair |= signature.executor_surface.requires_import_repair;
    state.executor_surfaces.requires_module_shim |= signature.executor_surface.requires_module_shim;
    state.executor_surfaces.requires_re_export_enforcement |= signature.executor_surface.requires_re_export_enforcement;
    state.executor_surfaces.requires_verification_gate |= signature.executor_surface.requires_verification_gate;

    // Increment action count
    state.action_count += 1;
}

/// Collect invariants that are touched by a signature
fn collect_invariants_touched(touchpoints: &InvariantTouchpoints) -> Vec<InvariantType> {
    let mut invariants = Vec::new();

    if touchpoints.i1_module_coherence {
        invariants.push(InvariantType::I1ModuleCoherence);
    }
    if touchpoints.i2_dependency_direction {
        invariants.push(InvariantType::I2DependencyDirection);
    }
    if touchpoints.visibility_law {
        invariants.push(InvariantType::VisibilityLaw);
    }
    if touchpoints.re_export_law {
        invariants.push(InvariantType::ReExportLaw);
    }
    if touchpoints.test_topology_law {
        invariants.push(InvariantType::TestTopologyLaw);
    }

    invariants
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a minimal valid signature for testing
    fn create_test_signature(
        action_id: &str,
        file_writes: Vec<PathBuf>,
        invariants: InvariantTouchpoints,
    ) -> EffectSignature {
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
                files: file_writes.into_iter().collect(),
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
            invariant_touchpoints: invariants,
            executor_surface: ExecutorSurface {
                requires_import_repair: false,
                requires_module_shim: false,
                requires_re_export_enforcement: false,
                requires_verification_gate: false,
            },
        }
    }

    #[test]
    fn test_empty_batch_is_admissible() {
        let result = compose_batch(&[]);
        match result {
            CompositionResult::Admissible { action_count, .. } => {
                assert_eq!(action_count, 0);
            }
            _ => panic!("Empty batch should be admissible"),
        }
    }

    #[test]
    fn test_single_action_is_admissible() {
        let sig = create_test_signature(
            "action1",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let result = compose_batch(&[sig]);
        match result {
            CompositionResult::Admissible { action_count, .. } => {
                assert_eq!(action_count, 1);
            }
            _ => panic!("Single action should be admissible"),
        }
    }

    #[test]
    fn test_non_overlapping_actions_are_admissible() {
        let sig1 = create_test_signature(
            "action1",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let sig2 = create_test_signature(
            "action2",
            vec![PathBuf::from("file2.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let result = compose_batch(&[sig1, sig2]);
        match result {
            CompositionResult::Admissible { action_count, .. } => {
                assert_eq!(action_count, 2);
            }
            _ => panic!("Non-overlapping actions should be admissible"),
        }
    }

    #[test]
    fn test_file_write_conflict_aborts() {
        let sig1 = create_test_signature(
            "action1",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let sig2 = create_test_signature(
            "action2",
            vec![PathBuf::from("file1.rs")], // Same file
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let result = compose_batch(&[sig1, sig2]);
        match result {
            CompositionResult::Inadmissible {
                first_failure_index,
                failed_action_id,
                conflict_reason,
                ..
            } => {
                assert_eq!(first_failure_index, 1);
                assert_eq!(failed_action_id, "action2");
                match conflict_reason {
                    ConflictReason::FileWriteConflict { prior_action_index, .. } => {
                        assert_eq!(prior_action_index, 0);
                    }
                    _ => panic!("Expected FileWriteConflict"),
                }
            }
            _ => panic!("Expected inadmissible due to file write conflict"),
        }
    }

    #[test]
    fn test_invariant_overlap_aborts() {
        let sig1 = create_test_signature(
            "action1",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: true, // Touches I1
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let sig2 = create_test_signature(
            "action2",
            vec![PathBuf::from("file2.rs")], // Different file
            InvariantTouchpoints {
                i1_module_coherence: true, // Also touches I1
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let result = compose_batch(&[sig1, sig2]);
        match result {
            CompositionResult::Inadmissible {
                first_failure_index,
                conflict_reason,
                ..
            } => {
                assert_eq!(first_failure_index, 1);
                match conflict_reason {
                    ConflictReason::InvariantOverlap { invariant, prior_action_index } => {
                        assert_eq!(invariant, InvariantType::I1ModuleCoherence);
                        assert_eq!(prior_action_index, 0);
                    }
                    _ => panic!("Expected InvariantOverlap"),
                }
            }
            _ => panic!("Expected inadmissible due to invariant overlap"),
        }
    }

    #[test]
    fn test_read_after_write_aborts() {
        let file_path = PathBuf::from("file1.rs");

        let sig1 = create_test_signature(
            "action1",
            vec![file_path.clone()], // Writes file1.rs
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let mut sig2 = create_test_signature(
            "action2",
            vec![PathBuf::from("file2.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );
        // Make sig2 read file1.rs
        sig2.reads.paths.insert(file_path.clone());

        let result = compose_batch(&[sig1, sig2]);
        match result {
            CompositionResult::Inadmissible {
                first_failure_index,
                conflict_reason,
                ..
            } => {
                assert_eq!(first_failure_index, 1);
                match conflict_reason {
                    ConflictReason::ReadAfterWriteAmbiguity { written_by_index, .. } => {
                        assert_eq!(written_by_index, 0);
                    }
                    _ => panic!("Expected ReadAfterWriteAmbiguity"),
                }
            }
            _ => panic!("Expected inadmissible due to read-after-write"),
        }
    }

    #[test]
    fn test_order_sensitivity() {
        // A reads file1, B writes file1
        // Order [A, B] should succeed (read before write is ok)
        // Order [B, A] should fail (read after write is conflict)

        let file_path = PathBuf::from("file1.rs");

        let mut sig_read = create_test_signature(
            "reader",
            vec![],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );
        sig_read.reads.paths.insert(file_path.clone());

        let sig_write = create_test_signature(
            "writer",
            vec![file_path.clone()],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        // Order [read, write] should be admissible
        let result1 = compose_batch(&[sig_read.clone(), sig_write.clone()]);
        assert!(
            matches!(result1, CompositionResult::Admissible { .. }),
            "Read before write should be admissible"
        );

        // Order [write, read] should be inadmissible
        let result2 = compose_batch(&[sig_write, sig_read]);
        assert!(
            matches!(result2, CompositionResult::Inadmissible { .. }),
            "Read after write should be inadmissible"
        );
    }

    #[test]
    fn test_executor_surface_accumulation() {
        let mut sig1 = create_test_signature(
            "action1",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );
        sig1.executor_surface.requires_import_repair = true;

        let mut sig2 = create_test_signature(
            "action2",
            vec![PathBuf::from("file2.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );
        sig2.executor_surface.requires_verification_gate = true;

        let result = compose_batch(&[sig1, sig2]);
        match result {
            CompositionResult::Admissible { final_state, .. } => {
                assert!(final_state.executor_surfaces.requires_import_repair);
                assert!(final_state.executor_surfaces.requires_verification_gate);
                assert!(!final_state.executor_surfaces.requires_module_shim);
            }
            _ => panic!("Expected admissible with accumulated executor surfaces"),
        }
    }

    #[test]
    fn test_abort_happens_at_first_conflict() {
        // Three actions: A writes file1, B writes file1 (conflict!), C writes file1
        // Should abort at B (index 1), never evaluate C

        let sig_a = create_test_signature(
            "action_a",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let sig_b = create_test_signature(
            "action_b",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let sig_c = create_test_signature(
            "action_c",
            vec![PathBuf::from("file1.rs")],
            InvariantTouchpoints {
                i1_module_coherence: false,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        );

        let result = compose_batch(&[sig_a, sig_b, sig_c]);
        match result {
            CompositionResult::Inadmissible {
                first_failure_index,
                failed_action_id,
                state_before_failure,
                ..
            } => {
                assert_eq!(first_failure_index, 1);
                assert_eq!(failed_action_id, "action_b");
                // State should only contain action_a
                assert_eq!(state_before_failure.action_count, 1);
            }
            _ => panic!("Expected inadmissible at index 1"),
        }
    }
}
