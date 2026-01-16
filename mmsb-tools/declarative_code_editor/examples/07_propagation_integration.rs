//! Example: Propagation Integration
//!
//! Demonstrates:
//! 1. Converting SourceBuffer → PageIndex
//! 2. Translating EditIntent for propagation
//! 3. The propagation flow (simplified without live DAG)

use declarative_code_editor::{
    source::SourceBuffer, source_buffer_to_page_index,
    EditIntent,
};
use std::path::PathBuf;

fn main() {
    println!("=== Propagation Integration Example ===\n");
    
    // Create root source file
    let root_src = r#"
pub struct Config {
    pub timeout: u64,
}

pub fn create_config() -> Config {
    Config { timeout: 30 }
}
"#;
    
    let root_buffer = SourceBuffer::new(
        PathBuf::from("src/config.rs"),
        root_src.to_string()
    ).unwrap();
    
    // Step 1: Convert SourceBuffer → PageIndex
    println!("Step 1: Converting SourceBuffer → PageIndex");
    let root_index = source_buffer_to_page_index(&root_buffer);
    println!("  Page ID: {}", root_index.page_id.0);
    println!("  Exports: {:?}", root_index.exports);
    println!("  Imports: {:?}", root_index.imports);
    println!("  References: {:?}", root_index.references);
    println!();
    
    // Step 2: Create intent for propagation
    println!("Step 2: Creating EditIntent");
    let intents = vec![EditIntent::RenameSymbol {
        old: "Config".to_string(),
        new: "Settings".to_string(),
    }];
    println!("  Intent: RenameSymbol {{ old: Config, new: Settings }}");
    println!();
    
    // Step 3: Translate intent (done internally by propagate_from_buffer)
    println!("Step 3: Intent Translation");
    println!("  declarative_code_editor::EditIntent::RenameSymbol");
    println!("  → structural_code_editor::EditIntent::RenameSymbol");
    println!();
    
    // Step 4: Propagation flow explanation
    println!("Step 4: Propagation Flow (in real usage):");
    println!("  1. Root edit triggers intent extraction");
    println!("  2. Intent translated to structural format");
    println!("  3. DependencyGraph queried for affected pages");
    println!("  4. Each affected page:");
    println!("     - Check if page imports/references renamed symbol");
    println!("     - If yes, rewrite page source with new symbol name");
    println!("     - Generate Delta for updated page");
    println!("  5. Return vector of PropagatedDelta");
    println!();
    
    println!("=== Integration Points ===");
    println!("✓ SourceBuffer → PageIndex conversion implemented");
    println!("✓ EditIntent translation implemented");
    println!("✓ propagate_from_buffer() ready for DAG integration");
    println!("✓ Rewrite logic handles symbol renaming and deletion");
    println!();
    
    println!("=== Next Steps ===");
    println!("In real usage:");
    println!("1. MMSB authority system maintains DependencyGraph");
    println!("2. After commit, MMSB calls propagate_from_buffer()");
    println!("3. Propagation engine generates derived deltas");
    println!("4. Derived deltas committed back to MMSB");
}
