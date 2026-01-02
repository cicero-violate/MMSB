//! PHASE 6.5 Admission Integration Demonstration
//!
//! This example demonstrates that admit_batch correctly:
//! 1. Detects conflicts (inadmissible batches)
//! 2. Allows valid batches (admissible batches)
//! 3. Always generates admission_composition.json
//! 4. Produces auditable proof artifacts

use mmsb_analyzer::{
    admit_batch, AdmissionDecision, EffectSignature, ExecutorSurface, InvariantTouchpoints,
    ReadEffects, StructuralTransitions, WriteEffects, SCHEMA_VERSION,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn create_action(action_id: &str, file_write: &str) -> EffectSignature {
    let mut files = BTreeSet::new();
    files.insert(PathBuf::from(file_write));

    EffectSignature {
        schema_version: SCHEMA_VERSION.to_string(),
        action_type: "MoveToLayer".to_string(),
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
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PHASE 6.5 Admission Integration Demonstration ===\n");

    // Test 1: Inadmissible batch (conflict)
    println!("Test 1: Inadmissible Batch (File Write Conflict)");
    println!("--------------------------------------------------");

    let action_a = create_action("action_a", "src/foo.rs");
    let action_b = create_action("action_b", "src/foo.rs"); // Same file - conflict!

    let batch_1 = vec![action_a.clone(), action_b];
    let artifact_path_1 = PathBuf::from("admission_composition_test1.json");

    println!("Batch: [action_a writes src/foo.rs, action_b writes src/foo.rs]");
    println!("Running admit_batch...\n");

    let decision_1 = admit_batch(&batch_1, &artifact_path_1)?;

    match &decision_1 {
        AdmissionDecision::Inadmissible { artifact_path } => {
            println!("✓ Result: INADMISSIBLE");
            println!("✓ Artifact written to: {}", artifact_path.display());
        }
        AdmissionDecision::Admissible { .. } => {
            println!("✗ ERROR: Expected inadmissible, got admissible!");
            return Err("Test failed".into());
        }
    }

    // Read and display artifact
    let artifact_json_1 = std::fs::read_to_string(&artifact_path_1)?;
    let artifact_1: serde_json::Value = serde_json::from_str(&artifact_json_1)?;

    println!("\nArtifact Content:");
    println!("  admissible: {}", artifact_1["admissible"]);
    println!("  batch_size: {}", artifact_1["batch_size"]);
    println!(
        "  first_failure_index: {}",
        artifact_1["composition_result"]["first_failure_index"]
    );
    println!(
        "  failed_action_id: {}",
        artifact_1["composition_result"]["failed_action_id"]
    );
    println!(
        "  conflict_type: {}",
        artifact_1["composition_result"]["conflict_reason"]["conflict_type"]
    );
    println!(
        "  conflict_file: {}",
        artifact_1["composition_result"]["conflict_reason"]["file"]
    );
    println!(
        "  prior_action_index: {}",
        artifact_1["composition_result"]["conflict_reason"]["prior_action_index"]
    );

    println!("\n{}", "=".repeat(60));
    println!();

    // Test 2: Admissible batch (no conflict)
    println!("Test 2: Admissible Batch (Different Files & Invariants)");
    println!("--------------------------------------------------------");

    // Create action_c with different invariants to avoid conservative conflict
    let mut action_c = create_action("action_c", "src/bar.rs"); // Different file
    action_c.invariant_touchpoints = InvariantTouchpoints {
        i1_module_coherence: false,
        i2_dependency_direction: false,
        visibility_law: true, // Different invariant
        re_export_law: false,
        test_topology_law: false,
    };

    let batch_2 = vec![action_a, action_c];
    let artifact_path_2 = PathBuf::from("admission_composition_test2.json");

    println!("Batch: [action_a writes src/foo.rs, action_c writes src/bar.rs]");
    println!("Running admit_batch...\n");

    let decision_2 = admit_batch(&batch_2, &artifact_path_2)?;

    match &decision_2 {
        AdmissionDecision::Admissible { artifact_path } => {
            println!("✓ Result: ADMISSIBLE");
            println!("✓ Artifact written to: {}", artifact_path.display());
        }
        AdmissionDecision::Inadmissible { .. } => {
            println!("✗ ERROR: Expected admissible, got inadmissible!");
            return Err("Test failed".into());
        }
    }

    // Read and display artifact
    let artifact_json_2 = std::fs::read_to_string(&artifact_path_2)?;
    let artifact_2: serde_json::Value = serde_json::from_str(&artifact_json_2)?;

    println!("\nArtifact Content:");
    println!("  admissible: {}", artifact_2["admissible"]);
    println!("  batch_size: {}", artifact_2["batch_size"]);
    println!(
        "  action_count: {}",
        artifact_2["composition_result"]["action_count"]
    );
    println!("  action_ids: {}", artifact_2["composition_result"]["action_ids"]);
    println!(
        "  requires_import_repair: {}",
        artifact_2["composition_result"]["executor_surfaces"]["requires_import_repair"]
    );
    println!(
        "  requires_module_shim: {}",
        artifact_2["composition_result"]["executor_surfaces"]["requires_module_shim"]
    );
    println!(
        "  requires_verification_gate: {}",
        artifact_2["composition_result"]["executor_surfaces"]["requires_verification_gate"]
    );
    println!(
        "  invariants_touched: {}",
        artifact_2["composition_result"]["invariants_touched"]
    );

    println!("\n{}", "=".repeat(60));
    println!("\n✓ All tests passed!");
    println!("\nPHASE 6.5 admission integration is working correctly:");
    println!("  - Conflict detection: ✓");
    println!("  - Valid batch acceptance: ✓");
    println!("  - Artifact generation: ✓");
    println!("  - Proof surface complete: ✓");

    // Note: Artifacts left in place for inspection
    println!("\nArtifact files:");
    println!("  - {}", artifact_path_1.display());
    println!("  - {}", artifact_path_2.display());

    Ok(())
}
