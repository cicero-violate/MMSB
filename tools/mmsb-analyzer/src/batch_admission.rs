//! Batch-Level Admission Predicate (BLAP) - PHASE 6.5 Final Component
//!
//! # ARCHITECTURAL FREEZE
//!
//! PHASE 6.5 Admission Intelligence is complete.
//! No changes permitted without explicit architectural authorization.
//!
//! # Purpose
//!
//! Provides the single admission decision API between correction intelligence and execution.
//! This is a thin, deterministic wrapper with no new logic.
//!
//! # Guarantees
//!
//! - **Always** runs composition fold
//! - **Always** generates admission_composition.json
//! - **Never** short-circuits on success or failure
//! - **Deterministic**: Same input → same decision + same artifact
//! - **Pure wrapper**: No logic beyond composition + artifact generation
//!
//! # Architecture
//!
//! Admission = Composition + Proof Artifact
//!
//! This is the **only gate** between correction intelligence and execution.
//!
//! # Version
//!
//! - 0.1.0: Initial implementation (PHASE 6.5 completion)

use crate::admission_composition_artifact::{generate_artifact, write_artifact};
use crate::composition_rule::compose_batch;
use crate::effect_signature_schema::EffectSignature;
use std::path::{Path, PathBuf};

/// Admission decision result
///
/// The artifact path is **always** populated. The artifact is the proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Batch is admissible (composition succeeded)
    Admissible {
        /// Path to generated artifact
        artifact_path: PathBuf,
    },
    /// Batch is inadmissible (composition failed)
    Inadmissible {
        /// Path to generated artifact (contains failure proof)
        artifact_path: PathBuf,
    },
}

impl AdmissionDecision {
    /// Check if batch was admissible
    pub fn is_admissible(&self) -> bool {
        matches!(self, AdmissionDecision::Admissible { .. })
    }

    /// Get artifact path (always present)
    pub fn artifact_path(&self) -> &Path {
        match self {
            AdmissionDecision::Admissible { artifact_path } => artifact_path,
            AdmissionDecision::Inadmissible { artifact_path } => artifact_path,
        }
    }
}

/// Admit a batch of actions with compositional proof
///
/// # Process (Invariant)
///
/// 1. Run `compose_batch(batch)` (always)
/// 2. Generate `admission_composition.json` (always)
/// 3. Write artifact to `artifact_path` (always)
/// 4. Return decision based on composition result
///
/// # Guarantees
///
/// - **No short-circuit**: Always completes full process
/// - **No interpretation**: Pure wrapper around composition + artifact
/// - **No branching**: Executor remains ignorant of this decision
/// - **Deterministic**: Same batch → same artifact → same decision
///
/// # Arguments
///
/// * `batch` - Ordered sequence of effect signatures
/// * `artifact_path` - Where to write admission_composition.json
///
/// # Returns
///
/// * `Ok(AdmissionDecision)` - Decision + artifact path
/// * `Err(...)` - Only on I/O failure writing artifact
pub fn admit_batch(
    batch: &[EffectSignature],
    artifact_path: &Path,
) -> Result<AdmissionDecision, Box<dyn std::error::Error>> {
    // Step 1: Always run composition fold
    let result = compose_batch(batch);

    // Step 2: Always generate artifact
    let artifact = generate_artifact(batch, &result);

    // Step 3: Always write artifact
    write_artifact(&artifact, artifact_path)?;

    // Step 4: Return decision (artifact is the proof)
    let decision = if artifact.admissible {
        AdmissionDecision::Admissible {
            artifact_path: artifact_path.to_path_buf(),
        }
    } else {
        AdmissionDecision::Inadmissible {
            artifact_path: artifact_path.to_path_buf(),
        }
    };

    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect_signature_schema::*;
    use std::collections::BTreeSet;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn create_test_signature(
        action_id: &str,
        file_writes: Vec<PathBuf>,
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
                requires_verification_gate: true,
            },
        }
    }

    #[test]
    fn test_admissible_batch() {
        let temp_dir = TempDir::new().unwrap();
        let artifact_path = temp_dir.path().join("admission_composition.json");

        // Create admissible batch (different files)
        let batch = vec![
            create_test_signature("action1", vec![PathBuf::from("file1.rs")]),
            create_test_signature("action2", vec![PathBuf::from("file2.rs")]),
        ];

        let decision = admit_batch(&batch, &artifact_path).unwrap();

        // Verify decision
        assert!(decision.is_admissible(), "Batch should be admissible");
        assert_eq!(decision.artifact_path(), artifact_path);

        // Verify artifact was written
        assert!(artifact_path.exists(), "Artifact file must exist");

        // Verify artifact content
        let artifact_json = std::fs::read_to_string(&artifact_path).unwrap();
        let artifact: serde_json::Value = serde_json::from_str(&artifact_json).unwrap();
        assert_eq!(artifact["admissible"], true);
        assert_eq!(artifact["batch_size"], 2);
    }

    #[test]
    fn test_inadmissible_batch() {
        let temp_dir = TempDir::new().unwrap();
        let artifact_path = temp_dir.path().join("admission_composition.json");

        // Create inadmissible batch (same file = conflict)
        let batch = vec![
            create_test_signature("action1", vec![PathBuf::from("file1.rs")]),
            create_test_signature("action2", vec![PathBuf::from("file1.rs")]),
        ];

        let decision = admit_batch(&batch, &artifact_path).unwrap();

        // Verify decision
        assert!(!decision.is_admissible(), "Batch should be inadmissible");
        assert_eq!(decision.artifact_path(), artifact_path);

        // Verify artifact was written
        assert!(artifact_path.exists(), "Artifact file must exist");

        // Verify artifact content
        let artifact_json = std::fs::read_to_string(&artifact_path).unwrap();
        let artifact: serde_json::Value = serde_json::from_str(&artifact_json).unwrap();
        assert_eq!(artifact["admissible"], false);
        assert_eq!(artifact["batch_size"], 2);

        // Verify failure details present
        let result = &artifact["composition_result"];
        assert_eq!(result["type"], "Inadmissible");
        assert_eq!(result["first_failure_index"], 1);
    }
}
