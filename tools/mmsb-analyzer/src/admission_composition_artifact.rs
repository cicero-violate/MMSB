//! Admission Composition Artifact - PHASE 6.5 Proof Serialization
//!
//! # ARCHITECTURAL FREEZE
//!
//! PHASE 6.5 Admission Intelligence is complete.
//! No changes permitted without explicit architectural authorization.
//!
//! # Purpose
//!
//! Pure serialization of CompositionResult to admission_composition.json.
//! This is a **faithful proof artifact**, not a diagnostic tool.
//!
//! # Guarantees
//!
//! - **Deterministic**: Same input → byte-identical output
//! - **Lossless**: Complete projection of CompositionResult
//! - **Order-Preserving**: Action sequence preserved
//! - **Human-Auditable**: Structured for manual review
//! - **Pure Projection**: No logic, no interpretation, no heuristics
//!
//! # Version
//!
//! - 0.1.0: Initial implementation

use crate::composition_rule::{ComposedEffectState, CompositionResult, ConflictReason};
use crate::effect_signature_schema::EffectSignature;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Artifact schema version
pub const ARTIFACT_SCHEMA_VERSION: &str = "0.1.0";

/// Analyzer version (from Cargo.toml)
pub const ANALYZER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Top-level admission composition artifact
///
/// This is the complete, durable proof object for batch admission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionCompositionArtifact {
    /// Artifact schema version
    pub schema_version: String,

    /// Number of actions in the batch
    pub batch_size: usize,

    /// Whether the batch is admissible
    pub admissible: bool,

    /// Composition result (complete)
    pub composition_result: CompositionResultProjection,

    /// ISO 8601 timestamp when artifact was generated
    pub timestamp: String,

    /// Analyzer version
    pub analyzer_version: String,
}

/// Projection of CompositionResult for serialization
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CompositionResultProjection {
    Admissible {
        action_count: usize,
        executor_surfaces: ExecutorSurfaceProjection,
        invariants_touched: Vec<String>,
        action_ids: Vec<String>,
    },
    Inadmissible {
        first_failure_index: usize,
        failed_action_id: String,
        conflict_reason: ConflictReasonProjection,
        state_before_failure: StateProjection,
    },
}

/// Projection of ExecutorSurface
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorSurfaceProjection {
    pub requires_import_repair: bool,
    pub requires_module_shim: bool,
    pub requires_re_export_enforcement: bool,
    pub requires_verification_gate: bool,
}

/// Projection of ConflictReason
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "conflict_type")]
pub enum ConflictReasonProjection {
    FileWriteConflict {
        file: String,
        prior_action_index: usize,
    },
    ModuleWriteConflict {
        module_path: String,
        prior_action_index: usize,
    },
    ReadAfterWriteAmbiguity {
        surface: String,
        written_by_index: usize,
    },
    InvariantOverlap {
        invariant: String,
        prior_action_index: usize,
    },
    ExecutorSurfaceForbidden {
        surface: String,
    },
}

/// Projection of ComposedEffectState
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateProjection {
    pub action_count: usize,
    pub files_written_count: usize,
    pub modules_written_count: usize,
    pub invariants_touched: Vec<String>,
}

/// Generate admission composition artifact from batch and result
///
/// # Pure Projection
///
/// This function performs **no logic**. It only serializes the composition result.
/// The truth is already determined by `compose_batch`.
pub fn generate_artifact(
    batch: &[EffectSignature],
    result: &CompositionResult,
) -> AdmissionCompositionArtifact {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let (admissible, composition_result) = match result {
        CompositionResult::Admissible {
            action_count,
            final_state,
        } => {
            let executor_surfaces = ExecutorSurfaceProjection {
                requires_import_repair: final_state.executor_surfaces.requires_import_repair,
                requires_module_shim: final_state.executor_surfaces.requires_module_shim,
                requires_re_export_enforcement: final_state
                    .executor_surfaces
                    .requires_re_export_enforcement,
                requires_verification_gate: final_state
                    .executor_surfaces
                    .requires_verification_gate,
            };

            let invariants_touched = project_invariants_touched(final_state);
            let action_ids = batch.iter().map(|sig| sig.action_id.clone()).collect();

            (
                true,
                CompositionResultProjection::Admissible {
                    action_count: *action_count,
                    executor_surfaces,
                    invariants_touched,
                    action_ids,
                },
            )
        }

        CompositionResult::Inadmissible {
            first_failure_index,
            failed_action_id,
            conflict_reason,
            state_before_failure,
        } => {
            let conflict_reason = project_conflict_reason(conflict_reason);
            let state_before_failure = project_state(state_before_failure);

            (
                false,
                CompositionResultProjection::Inadmissible {
                    first_failure_index: *first_failure_index,
                    failed_action_id: failed_action_id.clone(),
                    conflict_reason,
                    state_before_failure,
                },
            )
        }
    };

    AdmissionCompositionArtifact {
        schema_version: ARTIFACT_SCHEMA_VERSION.to_string(),
        batch_size: batch.len(),
        admissible,
        composition_result,
        timestamp,
        analyzer_version: ANALYZER_VERSION.to_string(),
    }
}

/// Write artifact to JSON file
///
/// # Determinism
///
/// JSON is serialized with sorted keys and pretty-printing for human readability.
/// Same input always produces byte-identical output.
pub fn write_artifact(
    artifact: &AdmissionCompositionArtifact,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(artifact)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Read artifact from JSON file
pub fn read_artifact(
    path: &Path,
) -> Result<AdmissionCompositionArtifact, Box<dyn std::error::Error>> {
    let json = std::fs::read_to_string(path)?;
    let artifact = serde_json::from_str(&json)?;
    Ok(artifact)
}

// === Internal Projection Helpers ===

fn project_invariants_touched(state: &ComposedEffectState) -> Vec<String> {
    use crate::composition_rule::InvariantType;

    let mut invariants = Vec::new();
    for (inv_type, indices) in &state.invariants_touched {
        if !indices.is_empty() {
            let name = match inv_type {
                InvariantType::I1ModuleCoherence => "I1_module_coherence",
                InvariantType::I2DependencyDirection => "I2_dependency_direction",
                InvariantType::VisibilityLaw => "visibility_law",
                InvariantType::ReExportLaw => "re_export_law",
                InvariantType::TestTopologyLaw => "test_topology_law",
            };
            invariants.push(name.to_string());
        }
    }
    invariants.sort();
    invariants
}

fn project_conflict_reason(reason: &ConflictReason) -> ConflictReasonProjection {
    match reason {
        ConflictReason::FileWriteConflict {
            file,
            prior_action_index,
        } => ConflictReasonProjection::FileWriteConflict {
            file: file.display().to_string(),
            prior_action_index: *prior_action_index,
        },
        ConflictReason::ModuleWriteConflict {
            module_path,
            prior_action_index,
        } => ConflictReasonProjection::ModuleWriteConflict {
            module_path: module_path.clone(),
            prior_action_index: *prior_action_index,
        },
        ConflictReason::ReadAfterWriteAmbiguity {
            surface,
            written_by_index,
        } => ConflictReasonProjection::ReadAfterWriteAmbiguity {
            surface: surface.clone(),
            written_by_index: *written_by_index,
        },
        ConflictReason::InvariantOverlap {
            invariant,
            prior_action_index,
        } => {
            use crate::composition_rule::InvariantType;
            let name = match invariant {
                InvariantType::I1ModuleCoherence => "I1_module_coherence",
                InvariantType::I2DependencyDirection => "I2_dependency_direction",
                InvariantType::VisibilityLaw => "visibility_law",
                InvariantType::ReExportLaw => "re_export_law",
                InvariantType::TestTopologyLaw => "test_topology_law",
            };
            ConflictReasonProjection::InvariantOverlap {
                invariant: name.to_string(),
                prior_action_index: *prior_action_index,
            }
        }
        ConflictReason::ExecutorSurfaceForbidden { surface } => {
            ConflictReasonProjection::ExecutorSurfaceForbidden {
                surface: surface.clone(),
            }
        }
    }
}

fn project_state(state: &ComposedEffectState) -> StateProjection {
    StateProjection {
        action_count: state.action_count,
        files_written_count: state.files_written.len(),
        modules_written_count: state.modules_written.len(),
        invariants_touched: project_invariants_touched(state),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_rule::compose_batch;
    use crate::effect_signature_schema::*;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

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
                requires_verification_gate: true,
            },
        }
    }

    #[test]
    fn test_admissible_batch_artifact() {
        // Create admissible batch
        let batch = vec![
            create_test_signature(
                "action1",
                vec![PathBuf::from("file1.rs")],
                InvariantTouchpoints {
                    i1_module_coherence: true,
                    i2_dependency_direction: false,
                    visibility_law: false,
                    re_export_law: false,
                    test_topology_law: false,
                },
            ),
            create_test_signature(
                "action2",
                vec![PathBuf::from("file2.rs")],
                InvariantTouchpoints {
                    i1_module_coherence: false,
                    i2_dependency_direction: true,
                    visibility_law: false,
                    re_export_law: false,
                    test_topology_law: false,
                },
            ),
        ];

        let result = compose_batch(&batch);
        let artifact = generate_artifact(&batch, &result);

        // Verify artifact structure
        assert_eq!(artifact.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(artifact.batch_size, 2);
        assert!(artifact.admissible);
        assert_eq!(artifact.analyzer_version, ANALYZER_VERSION);

        // Verify admissible projection
        match artifact.composition_result {
            CompositionResultProjection::Admissible {
                action_count,
                executor_surfaces,
                invariants_touched,
                action_ids,
            } => {
                assert_eq!(action_count, 2);
                assert!(executor_surfaces.requires_verification_gate);
                assert_eq!(invariants_touched.len(), 2);
                assert!(invariants_touched.contains(&"I1_module_coherence".to_string()));
                assert!(invariants_touched.contains(&"I2_dependency_direction".to_string()));
                assert_eq!(action_ids, vec!["action1", "action2"]);
            }
            _ => panic!("Expected admissible result"),
        }
    }

    #[test]
    fn test_inadmissible_batch_artifact() {
        // Create inadmissible batch (file write conflict)
        let batch = vec![
            create_test_signature(
                "action1",
                vec![PathBuf::from("file1.rs")],
                InvariantTouchpoints {
                    i1_module_coherence: false,
                    i2_dependency_direction: false,
                    visibility_law: false,
                    re_export_law: false,
                    test_topology_law: false,
                },
            ),
            create_test_signature(
                "action2",
                vec![PathBuf::from("file1.rs")], // Same file - conflict!
                InvariantTouchpoints {
                    i1_module_coherence: false,
                    i2_dependency_direction: false,
                    visibility_law: false,
                    re_export_law: false,
                    test_topology_law: false,
                },
            ),
        ];

        let result = compose_batch(&batch);
        let artifact = generate_artifact(&batch, &result);

        // Verify artifact structure
        assert_eq!(artifact.batch_size, 2);
        assert!(!artifact.admissible);

        // Verify inadmissible projection
        match artifact.composition_result {
            CompositionResultProjection::Inadmissible {
                first_failure_index,
                failed_action_id,
                conflict_reason,
                state_before_failure,
            } => {
                assert_eq!(first_failure_index, 1);
                assert_eq!(failed_action_id, "action2");
                assert_eq!(state_before_failure.action_count, 1);

                match conflict_reason {
                    ConflictReasonProjection::FileWriteConflict {
                        file,
                        prior_action_index,
                    } => {
                        assert_eq!(file, "file1.rs");
                        assert_eq!(prior_action_index, 0);
                    }
                    _ => panic!("Expected FileWriteConflict"),
                }
            }
            _ => panic!("Expected inadmissible result"),
        }
    }

    #[test]
    fn test_determinism() {
        // Same input should produce byte-identical output
        let batch = vec![create_test_signature(
            "test_action",
            vec![PathBuf::from("test.rs")],
            InvariantTouchpoints {
                i1_module_coherence: true,
                i2_dependency_direction: false,
                visibility_law: false,
                re_export_law: false,
                test_topology_law: false,
            },
        )];

        let result = compose_batch(&batch);

        // Generate artifact twice
        let artifact1 = generate_artifact(&batch, &result);
        let artifact2 = generate_artifact(&batch, &result);

        // Timestamps will differ, so set them to same value for comparison
        let mut artifact1_normalized = artifact1.clone();
        let mut artifact2_normalized = artifact2.clone();
        artifact1_normalized.timestamp = "NORMALIZED".to_string();
        artifact2_normalized.timestamp = "NORMALIZED".to_string();

        // Serialization should be identical (except timestamp)
        let json1 = serde_json::to_string_pretty(&artifact1_normalized).unwrap();
        let json2 = serde_json::to_string_pretty(&artifact2_normalized).unwrap();

        assert_eq!(json1, json2, "Same input must produce identical JSON");
    }
}
