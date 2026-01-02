//! CI Admission Gate Tests
//!
//! These tests enforce that admission is a CI-stopping event.
//! They must pass for CI to succeed.
//!
//! # Purpose
//!
//! Verify that:
//! 1. Inadmissible batches are correctly rejected
//! 2. Admissible batches are correctly accepted
//! 3. Artifacts are always generated
//! 4. No execution involvement required

use mmsb_analyzer::{
    admit_batch, EffectSignature, ExecutorSurface, InvariantTouchpoints,
    ReadEffects, StructuralTransitions, WriteEffects, SCHEMA_VERSION,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use tempfile::TempDir;

fn create_action(action_id: &str, file_write: &str) -> EffectSignature {
    let mut files = BTreeSet::new();
    files.insert(PathBuf::from(file_write));

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
            requires_verification_gate: true,
        },
    }
}

#[test]
fn ci_gate_inadmissible_batch_must_fail() {
    // CI REQUIREMENT: Inadmissible batch must be rejected
    let temp_dir = TempDir::new().unwrap();
    let artifact_path = temp_dir.path().join("admission_composition.json");

    // Create batch with file write conflict
    let action_a = create_action("action_a", "src/foo.rs");
    let action_b = create_action("action_b", "src/foo.rs"); // Conflict!

    let batch = vec![action_a, action_b];
    let decision = admit_batch(&batch, &artifact_path).expect("admit_batch should not panic");

    // Assert: Must be inadmissible
    assert!(
        !decision.is_admissible(),
        "CI GATE FAILURE: Inadmissible batch was marked admissible"
    );

    // Assert: Artifact must exist
    assert!(
        artifact_path.exists(),
        "CI GATE FAILURE: Artifact not written for inadmissible batch"
    );

    // Assert: Artifact must show correct failure
    let artifact_json = std::fs::read_to_string(&artifact_path).unwrap();
    let artifact: serde_json::Value = serde_json::from_str(&artifact_json).unwrap();

    assert_eq!(
        artifact["admissible"], false,
        "CI GATE FAILURE: Artifact claims batch is admissible"
    );
    assert_eq!(
        artifact["composition_result"]["first_failure_index"], 1,
        "CI GATE FAILURE: Wrong failure index"
    );
}

#[test]
fn ci_gate_admissible_batch_must_succeed() {
    // CI REQUIREMENT: Admissible batch must be accepted
    let temp_dir = TempDir::new().unwrap();
    let artifact_path = temp_dir.path().join("admission_composition.json");

    // Create batch with no conflicts
    let action_a = create_action("action_a", "src/foo.rs");
    let action_b = create_action("action_b", "src/bar.rs"); // Different file

    let batch = vec![action_a, action_b];
    let decision = admit_batch(&batch, &artifact_path).expect("admit_batch should not panic");

    // Assert: Must be admissible
    assert!(
        decision.is_admissible(),
        "CI GATE FAILURE: Admissible batch was marked inadmissible"
    );

    // Assert: Artifact must exist
    assert!(
        artifact_path.exists(),
        "CI GATE FAILURE: Artifact not written for admissible batch"
    );

    // Assert: Artifact must show success
    let artifact_json = std::fs::read_to_string(&artifact_path).unwrap();
    let artifact: serde_json::Value = serde_json::from_str(&artifact_json).unwrap();

    assert_eq!(
        artifact["admissible"], true,
        "CI GATE FAILURE: Artifact claims batch is inadmissible"
    );
    assert_eq!(
        artifact["batch_size"], 2,
        "CI GATE FAILURE: Wrong batch size"
    );
}

#[test]
fn ci_gate_artifact_always_written() {
    // CI REQUIREMENT: Artifact must be written regardless of outcome
    let temp_dir = TempDir::new().unwrap();

    // Test 1: Inadmissible batch
    let artifact_path_1 = temp_dir.path().join("artifact1.json");
    let batch_1 = vec![
        create_action("a1", "file.rs"),
        create_action("a2", "file.rs"),
    ];
    admit_batch(&batch_1, &artifact_path_1).unwrap();
    assert!(
        artifact_path_1.exists(),
        "CI GATE FAILURE: Artifact not written for inadmissible batch"
    );

    // Test 2: Admissible batch
    let artifact_path_2 = temp_dir.path().join("artifact2.json");
    let batch_2 = vec![
        create_action("a1", "file1.rs"),
        create_action("a2", "file2.rs"),
    ];
    admit_batch(&batch_2, &artifact_path_2).unwrap();
    assert!(
        artifact_path_2.exists(),
        "CI GATE FAILURE: Artifact not written for admissible batch"
    );
}
