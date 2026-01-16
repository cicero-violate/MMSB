//! Example 1: Simple Function Rename
//!
//! Shows basic query + mutation to rename a function.

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Example 1: Simple Function Rename ===\n");

    let source = r#"
fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = calculate_sum(5, 3);
    println!("Sum: {}", result);
}
"#;

    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/math.rs"),
        source.to_string(),
    ).unwrap();

    println!("BEFORE:");
    println!("{}", buffer.source());

    // Query: Find function named "calculate_sum"
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("calculate_sum"));

    // Mutation: Rename to "add_numbers"
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "add_numbers"));

    // Execute through bridge
    let page_id = PageID(1);
    match BridgeOrchestrator::execute_and_bridge(&mut buffer, &mutation, page_id) {
        Ok(output) => {
            println!("\n✅ SUCCESS");
            println!("└─ Generated {} page delta(s)", output.page_deltas.len());
            println!("└─ Generated {} structural op(s)", output.structural_ops.len());
            println!("└─ Pipeline route: {:?}", output.route);
            
            if output.needs_state_commit() {
                println!("\n📋 Next Step: commit_delta() to TLog");
                println!("   This will trigger propagation to dependent pages");
            }
        }
        Err(e) => eprintln!("❌ Error: {:?}", e),
    }
}
