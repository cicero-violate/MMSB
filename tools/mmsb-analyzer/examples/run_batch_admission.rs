//! PHASE 6.5 Category 1 Wiring: Batch Admission Runner
//!
//! This binary wires batch-level admission into the analysis workflow.
//! It runs unconditionally after correction intelligence is generated.
//!
//! **Purpose**: Convert correction plans → effect signatures → admit_batch
//! **Output**: admission_composition.json (always generated)
//! **Constraints**: No interpretation, no suppression, no auto-execution

use mmsb_analyzer::{
    admit_batch, effect_signature_schema::ImportWrite, EffectSignature, ExecutorSurface,
    InvariantTouchpoints, ReadEffects, StructuralTransitions, WriteEffects, SCHEMA_VERSION,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: run_batch_admission <correction_intelligence.json> <output_admission.json>");
        process::exit(1);
    }

    let ci_path = &args[1];
    let output_path = &args[2];

    println!("🔬 PHASE 6.5 Batch Admission");
    println!("   Input: {}", ci_path);
    println!("   Output: {}", output_path);

    // Read correction intelligence
    let ci_json = match fs::read_to_string(ci_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!("Error reading correction intelligence: {}", e);
            process::exit(1);
        }
    };

    let ci: Value = match serde_json::from_str(&ci_json) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing correction intelligence JSON: {}", e);
            process::exit(1);
        }
    };

    // Extract correction plans
    let plans = match ci["correction_plans"].as_array() {
        Some(p) => p,
        None => {
            eprintln!("Error: correction_plans not found or not an array");
            process::exit(1);
        }
    };

    println!("   Plans: {} correction plans", plans.len());

    // Convert to effect signatures
    let mut signatures: Vec<EffectSignature> = Vec::new();
    for plan in plans {
        if let Some(sig) = plan_to_effect_signature(plan) {
            signatures.push(sig);
        }
    }

    println!("   Signatures: {} effect signatures generated", signatures.len());

    // Run batch admission
    let artifact_path = PathBuf::from(output_path);
    match admit_batch(&signatures, &artifact_path) {
        Ok(decision) => {
            if decision.is_admissible() {
                println!("   ✅ Batch is ADMISSIBLE");
            } else {
                println!("   ❌ Batch is INADMISSIBLE");
            }
            println!("   Artifact written: {}", output_path);
        }
        Err(e) => {
            eprintln!("Error running batch admission: {}", e);
            process::exit(1);
        }
    }
}

fn plan_to_effect_signature(plan: &Value) -> Option<EffectSignature> {
    let action_id = plan["action_id"].as_str()?;
    let strategies = plan["correction_strategies"].as_array()?;

    let mut files_written = BTreeSet::new();
    let mut files_read = BTreeSet::new();
    let mut modules_touched = BTreeSet::new();
    let mut imports_touched: BTreeSet<ImportWrite> = BTreeSet::new();

    let mut action_type = "Unknown".to_string();
    let mut touches_i1 = false;
    let mut touches_i2 = false;
    let mut touches_visibility = false;

    for strategy in strategies {
        let strategy_type = strategy["type"].as_str()?;
        action_type = strategy_type.to_string();

        match strategy_type {
            "UpdatePath" => {
                if let Some(old) = strategy["old_path"].as_str() {
                    files_read.insert(PathBuf::from(old));
                }
                if let Some(new) = strategy["new_path"].as_str() {
                    files_written.insert(PathBuf::from(new));
                }
                touches_i1 = true;
            }
            "UpdateCaller" => {
                if let Some(file) = strategy["caller_file"].as_str() {
                    files_written.insert(PathBuf::from(file));
                }
                touches_i1 = true;
            }
            "MoveToLayer" => {
                if let Some(from) = strategy.get("from_path").and_then(|v| v.as_str()) {
                    files_read.insert(PathBuf::from(from));
                }
                if let Some(to) = strategy.get("to_path").and_then(|v| v.as_str()) {
                    files_written.insert(PathBuf::from(to));
                }
                touches_i1 = true;
                touches_i2 = true;
            }
            "EnsureImports" => {
                if let Some(file) = strategy.get("target_file").and_then(|v| v.as_str()) {
                    files_written.insert(PathBuf::from(file));
                    imports_touched.insert(ImportWrite {
                        target_file: PathBuf::from(file),
                        import_path: "unknown".to_string(),
                        is_re_export: false,
                    });
                }
            }
            "VisibilityPlan" => {
                if let Some(file) = strategy.get("file").and_then(|v| v.as_str()) {
                    files_written.insert(PathBuf::from(file));
                }
                touches_visibility = true;
            }
            _ => {}
        }
    }

    Some(EffectSignature {
        schema_version: SCHEMA_VERSION.to_string(),
        action_type: action_type.clone(),
        action_id: action_id.to_string(),
        reads: ReadEffects {
            paths: files_read,
            symbols: BTreeSet::new(),
            visibility_scopes: BTreeSet::new(),
            module_boundaries: BTreeSet::new(),
        },
        writes: WriteEffects {
            files: files_written,
            modules: modules_touched,
            imports: imports_touched,
            re_exports: BTreeSet::new(),
            visibility_modifiers: BTreeSet::new(),
        },
        structural_transitions: StructuralTransitions {
            file_to_module: Vec::new(),
            module_to_layer: Vec::new(),
            test_boundary_crossings: Vec::new(),
        },
        invariant_touchpoints: InvariantTouchpoints {
            i1_module_coherence: touches_i1,
            i2_dependency_direction: touches_i2,
            visibility_law: touches_visibility,
            re_export_law: false,
            test_topology_law: false,
        },
        executor_surface: ExecutorSurface {
            requires_import_repair: action_type == "EnsureImports",
            requires_module_shim: action_type == "MoveToLayer",
            requires_re_export_enforcement: false,
            requires_verification_gate: true,
        },
    })
}
