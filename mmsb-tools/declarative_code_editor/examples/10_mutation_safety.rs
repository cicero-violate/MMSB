//! Example: Mutation Safety Features
//!
//! Demonstrates:
//! 1. Dry-run preview mode
//! 2. Conflict detection

use declarative_code_editor::{
    SourceBuffer, MutationPlan, QueryPlan,
    KindPredicate, NamePredicate, ItemKind,
    apply_mutation,
};
use declarative_code_editor::executor::{preview_mutation, ConflictDetector};
use declarative_code_editor::mutation::ReplaceOp;
use std::path::PathBuf;

fn main() {
    println!("=== Mutation Safety Features Example ===\n");
    
    demo_preview_mode();
    println!();
    demo_conflict_detection();
}

fn demo_preview_mode() {
    println!("--- 1. Dry-Run Preview Mode ---");
    
    let src = r#"
fn calculate(x: i32) -> i32 {
    x + 1
}

fn process(y: i32) -> i32 {
    y * 2
}
"#;
    
    let buffer = SourceBuffer::new(
        PathBuf::from("src/math.rs"),
        src.to_string()
    ).unwrap();
    
    println!("Original source:");
    println!("{}", buffer.source());
    
    // Create mutation plan
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("calculate"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("name", "compute"));
    
    // Preview without applying
    println!("Previewing mutation...");
    let preview = preview_mutation(&buffer, &mutation).unwrap();
    
    println!("\nPreview Results:");
    println!("  Matched {} items", preview.matched_items.len());
    for item in &preview.matched_items {
        println!("    - {} '{}' at line {}", item.kind, item.name, item.line_start);
    }
    println!("  Would modify: {} items\n", preview.would_modify);
    
    println!("After (preview only):");
    println!("{}", preview.source_after);
    
    println!("Note: Original buffer unchanged (dry-run mode)");
}

fn demo_conflict_detection() {
    println!("--- 2. Conflict Detection ---");
    
    let src = r#"
fn foo() -> i32 {
    42
}

fn bar() -> i32 {
    100
}
"#;
    
    let buffer = SourceBuffer::new(
        PathBuf::from("src/lib.rs"),
        src.to_string()
    ).unwrap();
    
    // Create two mutations targeting the same function
    let mutation1 = MutationPlan::new(
        QueryPlan::new()
            .with_predicate(KindPredicate::new(ItemKind::Function))
            .with_predicate(NamePredicate::new("foo"))
    ).with_operation(ReplaceOp::new("name", "first"));
    
    let mutation2 = MutationPlan::new(
        QueryPlan::new()
            .with_predicate(KindPredicate::new(ItemKind::Function))
            .with_predicate(NamePredicate::new("foo"))
    ).with_operation(ReplaceOp::new("name", "second"));
    
    let mut detector = ConflictDetector::new();
    
    println!("Checking mutation 1...");
    match detector.check_conflict(&buffer, &mutation1) {
        Ok(()) => println!("  ✓ No conflict"),
        Err(e) => println!("  ✗ Conflict: {}", e),
    }
    
    println!("Checking mutation 2 (targets same function)...");
    match detector.check_conflict(&buffer, &mutation2) {
        Ok(()) => println!("  ✓ No conflict"),
        Err(e) => println!("  ✗ Conflict detected: {}", e),
    }
    
    println!("\nResetting detector...");
    detector.reset();
    
    println!("Checking mutation 2 again after reset...");
    match detector.check_conflict(&buffer, &mutation2) {
        Ok(()) => println!("  ✓ No conflict (after reset)"),
        Err(e) => println!("  ✗ Conflict: {}", e),
    }
}
