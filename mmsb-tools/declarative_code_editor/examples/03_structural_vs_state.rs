//! Example 3: Structural vs State Changes
//!
//! Demonstrates the difference between:
//! - STATE changes (rename function body)
//! - STRUCTURAL changes (add/remove imports)

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Example 3: Structural vs State Changes ===\n");

    // Example showing both types of changes
    let source = r#"
use std::collections::HashMap;

fn process_map(data: HashMap<String, i32>) -> i32 {
    data.values().sum()
}
"#;

    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/lib.rs"),
        source.to_string(),
    ).unwrap();

    println!("SOURCE:");
    println!("{}", buffer.source());

    // STATE CHANGE: Rename function (doesn't change DAG)
    println!("\n--- STATE CHANGE Example ---");
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("process_map"));

    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "sum_values"));

    let page_id = PageID(200);
    match BridgeOrchestrator::execute_and_bridge(&mut buffer, &mutation, page_id) {
        Ok(output) => {
            println!("✅ Function renamed");
            println!("Pipeline: {:?}", output.route);
            println!("Page deltas: {}", output.page_deltas.len());
            println!("Structural ops: {}", output.structural_ops.len());
            
            println!("\n📊 Analysis:");
            println!("- This is a STATE change");
            println!("- Goes to: commit_delta() → TLog → Propagation");
            println!("- DAG is NOT modified");
            println!("- Dependent pages will be rewritten");
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }

    println!("\n--- STRUCTURAL CHANGE Example (Conceptual) ---");
    println!("If we were to ADD an import:");
    println!("  use std::collections::HashSet;");
    println!("\nThis would be a STRUCTURAL change:");
    println!("- Creates StructuralOp::AddEdge");
    println!("- From: this_page → To: std::collections (new dependency)");
    println!("- Goes to: ShadowGraph → validate → commit_structural_delta()");
    println!("- DAG is MODIFIED");
    println!("- Does NOT trigger propagation");
    
    println!("\n--- BOTH Example (Conceptual) ---");
    println!("If we RENAME a module:");
    println!("  mod old_name {{ }} → mod new_name {{ }}");
    println!("\nThis requires BOTH pipelines:");
    println!("1. STRUCTURAL: Update DAG node ID");
    println!("2. STATE: Update page content");
    println!("Order: Structural FIRST, then State");
}
