//! Example: Query Combinators and Advanced Predicates
//!
//! Demonstrates:
//! 1. AND, OR, NOT combinators
//! 2. GenericPredicate - match generic items
//! 3. SignaturePredicate - match function signatures

use declarative_code_editor::{
    SourceBuffer, QueryPlan, execute_query,
    KindPredicate, NamePredicate, ItemKind,
    VisibilityPredicate, AttributePredicate,
    and, or, not,
    GenericPredicate, SignaturePredicate,
};
use std::path::PathBuf;

fn main() {
    println!("=== Query Combinators & Advanced Predicates Example ===\n");
    
    let src = r#"
pub fn simple_function() -> i32 {
    42
}

pub async fn async_task(url: String) -> Result<String, Error> {
    fetch(url).await
}

pub fn generic_function<T>(value: T) -> T {
    value
}

#[derive(Debug)]
pub struct Point<T> {
    x: T,
    y: T,
}

#[derive(Clone)]
struct Config {
    timeout: u64,
}

pub struct NonGeneric {
    data: String,
}

const MAX: usize = 100;
"#;
    
    let buffer = SourceBuffer::new(
        PathBuf::from("src/lib.rs"),
        src.to_string()
    ).unwrap();
    
    demo_and_combinator(&buffer);
    println!();
    demo_or_combinator(&buffer);
    println!();
    demo_not_combinator(&buffer);
    println!();
    demo_generic_predicate(&buffer);
    println!();
    demo_signature_predicate(&buffer);
}

fn demo_and_combinator(buffer: &SourceBuffer) {
    println!("--- 1. AND Combinator: Public Functions ---");
    
    // Find functions that are both public AND functions
    let query = QueryPlan::new()
        .with_predicate(and(
            KindPredicate::new(ItemKind::Function),
            VisibilityPredicate::public()
        ));
    
    let results = execute_query(buffer, &query);
    println!("Found {} public functions:", results.len());
    for item in results {
        if let syn::Item::Fn(func) = item {
            println!("  - {}", func.sig.ident);
        }
    }
}

fn demo_or_combinator(buffer: &SourceBuffer) {
    println!("--- 2. OR Combinator: Functions OR Structs ---");
    
    // Find items that are either functions or structs
    let query = QueryPlan::new()
        .with_predicate(or(
            KindPredicate::new(ItemKind::Function),
            KindPredicate::new(ItemKind::Struct)
        ));
    
    let results = execute_query(buffer, &query);
    println!("Found {} functions or structs:", results.len());
    for item in results {
        match item {
            syn::Item::Fn(func) => println!("  - fn {}", func.sig.ident),
            syn::Item::Struct(s) => println!("  - struct {}", s.ident),
            _ => {}
        }
    }
}

fn demo_not_combinator(buffer: &SourceBuffer) {
    println!("--- 3. NOT Combinator: Non-Public Structs ---");
    
    // Find structs that are NOT public
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Struct))
        .with_predicate(not(VisibilityPredicate::public()));
    
    let results = execute_query(buffer, &query);
    println!("Found {} private structs:", results.len());
    for item in results {
        if let syn::Item::Struct(s) = item {
            println!("  - {}", s.ident);
        }
    }
}

fn demo_generic_predicate(buffer: &SourceBuffer) {
    println!("--- 4. GenericPredicate: Generic Items ---");
    
    // Find all items with generics
    let query = QueryPlan::new()
        .with_predicate(GenericPredicate::any());
    
    let results = execute_query(buffer, &query);
    println!("Found {} generic items:", results.len());
    for item in results {
        match item {
            syn::Item::Fn(func) => println!("  - fn {}", func.sig.ident),
            syn::Item::Struct(s) => println!("  - struct {}", s.ident),
            _ => {}
        }
    }
}

fn demo_signature_predicate(buffer: &SourceBuffer) {
    println!("--- 5. SignaturePredicate: Async Functions ---");
    
    // Find async functions
    let query = QueryPlan::new()
        .with_predicate(SignaturePredicate::new().is_async());
    
    let results = execute_query(buffer, &query);
    println!("Found {} async functions:", results.len());
    for item in results {
        if let syn::Item::Fn(func) = item {
            println!("  - {}", func.sig.ident);
        }
    }
    
    println!();
    println!("--- 6. SignaturePredicate: Functions with 1+ Parameters ---");
    
    // Find functions with at least one parameter
    let query = QueryPlan::new()
        .with_predicate(SignaturePredicate::new().min_params(1));
    
    let results = execute_query(buffer, &query);
    println!("Found {} functions with params:", results.len());
    for item in results {
        if let syn::Item::Fn(func) = item {
            println!("  - {} ({} params)", func.sig.ident, func.sig.inputs.len());
        }
    }
}
