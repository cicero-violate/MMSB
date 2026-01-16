//! Bridge Example
//!
//! Demonstrates the complete flow from declarative edits to MMSB authority pipelines.
//!
//! Flow:
//! 1. Create declarative mutation (query + mutate)
//! 2. Execute through BridgeOrchestrator
//! 3. Receive BridgedOutput with:
//!    - Extracted semantic intents
//!    - Page deltas (STATE PIPELINE)
//!    - Structural ops (STRUCTURAL PIPELINE)
//!    - Pipeline routing decision

use declarative_code_editor::*;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Bridge Example: Declarative Edit → MMSB Pipelines ===\n");

    // Sample Rust code
    let source = r#"
fn old_function_name(x: i32) -> i32 {
    x + 1
}

fn another_function() {
    old_function_name(5);
}
"#;

    // Create buffer
    let mut buffer = EditBuffer::new(source.to_string());
    let page_id = PageID(12345);
    let file_path = PathBuf::from("src/example.rs");

    // Build query: find function named "old_function_name"
    let query = QueryPlan::new()
        .with_predicate(query::ItemKind::Function)
        .with_predicate(NamePredicate::new("old_function_name"));

    // Build mutation: rename it
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new(
            "sig.ident",
            "new_function_name",
        ));

    println!("📝 Original source:");
    println!("{}", source);
    println!();

    // Execute through bridge orchestrator
    match BridgeOrchestrator::execute_and_bridge(
        &mutation,
        &mut buffer,
        page_id,
        &file_path,
        false, // allow_empty
        false, // allow_multiple
    ) {
        Ok(output) => {
            println!("✅ Bridge execution successful!\n");

            // Show extracted intents
            println!("🎯 Extracted Semantic Intents:");
            for intent in &output.intents {
                println!("   {:?}", intent);
            }
            println!();

            // Show pipeline routing
            println!("🚦 Pipeline Routing: {:?}", output.route);
            println!("   - Needs structural commit: {}", output.needs_structural_commit());
            println!("   - Needs state commit: {}", output.needs_state_commit());
            println!();

            // Show page deltas
            println!("📦 Page Deltas (STATE PIPELINE): {} delta(s)", output.page_deltas.len());
            for delta in &output.page_deltas {
                println!("   - DeltaID: {:?}, PageID: {:?}", delta.delta_id, delta.page_id);
            }
            println!();

            // Show structural ops
            println!("🔗 Structural Ops (STRUCTURAL PIPELINE): {} op(s)", output.structural_ops.len());
            for op in &output.structural_ops {
                println!("   - {:?}", op);
            }
            println!();

            // Show final source
            println!("📝 Modified source:");
            println!("{}", buffer.render());
            println!();

            // Next steps (conceptual)
            println!("📋 Next Steps (Authority Model):");
            println!("   1. If structural ops exist:");
            println!("      → Build ShadowGraph from ops");
            println!("      → Validate (acyclic, references exist)");
            println!("      → Require JudgmentToken (structural)");
            println!("      → commit_structural_delta(ops, judgment)");
            println!();
            println!("   2. If page deltas exist:");
            println!("      → Require JudgmentToken (state)");
            println!("      → commit_delta(delta, judgment)");
            println!("      → Persist to TLog");
            println!("      → Snapshot DAG (read-only)");
            println!("      → Trigger propagation engine");
            println!("      → Emit derived deltas to dependents");
        }
        Err(e) => {
            eprintln!("❌ Bridge execution failed: {:?}", e);
        }
    }
}
