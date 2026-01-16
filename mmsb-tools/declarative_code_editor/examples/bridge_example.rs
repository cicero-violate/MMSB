use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Declarative Code Editor: Bridge Example ===\n");

    let source = r#"
fn process_data(x: i32) -> i32 {
    x + 1
}
"#;

    let page_id = PageID(12345);
    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/example.rs"),
        source.to_string(),
    ).unwrap();

    println!("📝 Original source:");
    println!("{}", buffer.source());
    println!();

    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("process_data"));

    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "transform_value"));

    match BridgeOrchestrator::execute_and_bridge(&mut buffer, &mutation, page_id) {
        Ok(output) => {
            println!("✅ Bridge execution successful!\n");
            println!("🎯 Intents: {} extracted", output.intents.len());
            println!("📦 Page Deltas: {}", output.page_deltas.len());
            println!("🔗 Structural Ops: {}", output.structural_ops.len());
            println!("🚦 Route: {:?}", output.route);
        }
        Err(e) => {
            eprintln!("❌ Error: {:?}", e);
        }
    }
}
