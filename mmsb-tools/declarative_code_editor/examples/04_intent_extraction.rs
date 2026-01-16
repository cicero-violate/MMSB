//! Example 4: Intent Extraction
//!
//! Shows how semantic intent is extracted from mutations.

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Example 4: Intent Extraction ===\n");

    let source = r#"
pub struct DataProcessor {
    buffer: Vec<u8>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }
    
    pub fn process(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }
}
"#;

    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/processor.rs"),
        source.to_string(),
    ).unwrap();

    println!("SOURCE:");
    println!("{}", buffer.source());

    // Rename struct
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Struct))
        .with_predicate(NamePredicate::new("DataProcessor"));

    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("ident", "StreamProcessor"));

    let page_id = PageID(300);
    match BridgeOrchestrator::execute_and_bridge(&mut buffer, &mutation, page_id) {
        Ok(output) => {
            println!("\n✅ Mutation applied");
            println!("Extracted {} intent(s)", output.intents.len());
            
            println!("\n📝 Intent Analysis:");
            if output.intents.is_empty() {
                println!("(Intent extraction not yet implemented)");
                println!("\nExpected intent:");
                println!("  EditIntent::RenameSymbol {{");
                println!("    old: \"DataProcessor\",");
                println!("    new: \"StreamProcessor\"");
                println!("  }}");
                
                println!("\n🔄 Propagation Impact:");
                println!("  Any page that references DataProcessor will be rewritten:");
                println!("  - use crate::DataProcessor → use crate::StreamProcessor");
                println!("  - let p = DataProcessor::new() → let p = StreamProcessor::new()");
            } else {
                for (i, intent) in output.intents.iter().enumerate() {
                    println!("  {}. {:?}", i + 1, intent);
                    println!("     Category: {:?}", intent.category());
                }
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
