//! Example 2: Multi-File Refactoring Simulation
//!
//! Shows how to handle multiple files with the declarative editor.
//! Demonstrates the pattern for cross-file changes.

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Example 2: Multi-File Refactoring ===\n");

    // File 1: Core module
    let core_source = r#"
pub fn process_data(input: &str) -> String {
    input.to_uppercase()
}

pub fn validate_input(input: &str) -> bool {
    !input.is_empty()
}
"#;

    // File 2: Uses core module
    let user_source = r#"
use crate::core::process_data;

fn handle_request(data: &str) {
    let processed = process_data(data);
    println!("Processed: {}", processed);
}
"#;

    let mut core_buffer = SourceBuffer::new(
        PathBuf::from("src/core.rs"),
        core_source.to_string(),
    ).unwrap();

    let mut user_buffer = SourceBuffer::new(
        PathBuf::from("src/user.rs"),
        user_source.to_string(),
    ).unwrap();

    println!("FILE 1 (src/core.rs) BEFORE:");
    println!("{}", core_buffer.source());
    println!("\nFILE 2 (src/user.rs) BEFORE:");
    println!("{}", user_buffer.source());

    // Rename in core module
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("process_data"));

    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "transform_data"));

    let core_page_id = PageID(100);
    let user_page_id = PageID(101);

    match BridgeOrchestrator::execute_and_bridge(&mut core_buffer, &mutation, core_page_id) {
        Ok(output) => {
            println!("\n✅ Core module updated");
            println!("└─ Delta for page {}", core_page_id.0);
            println!("└─ Route: {:?}", output.route);

            if output.needs_state_commit() {
                println!("\n📋 Propagation Flow:");
                println!("1. commit_delta(core_delta, judgment)");
                println!("2. TLog append → page {} updated", core_page_id.0);
                println!("3. Snapshot DAG (read-only)");
                println!("4. PropagationEngine finds dependents");
                println!("5. Rewrite page {} (user.rs):", user_page_id.0);
                println!("   - Old: use crate::core::process_data;");
                println!("   - New: use crate::core::transform_data;");
                println!("6. Emit derived delta for page {}", user_page_id.0);
                println!("7. commit_delta(user_delta, judgment)");
                
                println!("\n🔗 DAG Structure:");
                println!("   Page {} (core.rs) ← depends ← Page {} (user.rs)", 
                    core_page_id.0, user_page_id.0);
            }
        }
        Err(e) => eprintln!("❌ Error: {:?}", e),
    }
}
