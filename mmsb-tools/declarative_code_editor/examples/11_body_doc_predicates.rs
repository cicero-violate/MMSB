//! Example: Body and Doc Predicates
//!
//! Demonstrates:
//! 1. BodyPredicate - match functions by body content
//! 2. DocPredicate - match items by doc comments

use declarative_code_editor::{
    SourceBuffer, QueryPlan, execute_query,
    KindPredicate, ItemKind,
    BodyPredicate, DocPredicate,
};
use std::path::PathBuf;

fn main() {
    println!("=== Body and Doc Predicates Example ===\n");
    
    let src = r#"
/// Calculates the sum of two numbers
/// This is a simple addition function
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Deprecated function - use add() instead
#[deprecated]
pub fn sum(x: i32, y: i32) -> i32 {
    x + y
}

/// Multiplies two numbers together
pub fn multiply(x: i32, y: i32) -> i32 {
    let result = x * y;
    result
}

pub fn no_docs(val: i32) -> i32 {
    val + 10
}
"#;
    
    let buffer = SourceBuffer::new(
        PathBuf::from("src/math.rs"),
        src.to_string()
    ).unwrap();
    
    demo_body_predicate(&buffer);
    println!();
    demo_doc_predicate(&buffer);
}

fn demo_body_predicate(buffer: &SourceBuffer) {
    println!("--- 1. BodyPredicate: Functions containing 'result' ---");
    
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(BodyPredicate::contains("result"));
    
    let results = execute_query(buffer, &query);
    println!("Found {} functions with 'result' in body:", results.len());
    for item in results {
        if let syn::Item::Fn(func) = item {
            println!("  - {}", func.sig.ident);
        }
    }
}

fn demo_doc_predicate(buffer: &SourceBuffer) {
    println!("--- 2. DocPredicate: Items with 'Deprecated' in docs ---");
    
    let query = QueryPlan::new()
        .with_predicate(DocPredicate::contains("Deprecated"));
    
    let results = execute_query(buffer, &query);
    println!("Found {} items with 'Deprecated' in documentation:", results.len());
    for item in results {
        match item {
            syn::Item::Fn(func) => println!("  - fn {}", func.sig.ident),
            syn::Item::Struct(s) => println!("  - struct {}", s.ident),
            _ => {}
        }
    }
    
    println!();
    println!("--- 3. DocPredicate: Items mentioning 'sum' or 'add' ---");
    
    let query = QueryPlan::new()
        .with_predicate(DocPredicate::contains("sum"));
    
    let results = execute_query(buffer, &query);
    println!("Found {} items with 'sum' in docs:", results.len());
    for item in results {
        if let syn::Item::Fn(func) = item {
            println!("  - {}", func.sig.ident);
        }
    }
}
