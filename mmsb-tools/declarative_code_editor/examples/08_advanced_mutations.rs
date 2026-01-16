//! Example: Advanced Mutation Operations
//!
//! Demonstrates:
//! 1. ParamReplaceOp - replace function parameters
//! 2. AddParamOp - add new parameters
//! 3. RemoveParamOp - remove parameters
//! 4. BodyReplaceOp - replace expressions in function body

use declarative_code_editor::{
   SourceBuffer, MutationPlan, QueryPlan,
    KindPredicate, NamePredicate, ItemKind,
    apply_mutation,
};
use declarative_code_editor::mutation::{
    ParamReplaceOp, AddParamOp, RemoveParamOp, BodyReplaceOp,
};
use std::path::PathBuf;

fn main() {
    println!("=== Advanced Mutation Operations Example ===\n");
    
    demo_param_replace();
    println!();
    demo_add_param();
    println!();
    demo_remove_param();
    println!();
    demo_body_replace();
}

fn demo_param_replace() {
    println!("--- 1. ParamReplaceOp: Replace Parameter ---");
    
    let src = r#"
fn calculate(x: i32, y: i32) -> i32 {
    x + y
}
"#;
    
    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/math.rs"),
        src.to_string()
    ).unwrap();
    
   println!("Original:");
    println!("{}", buffer.source());
    
    // Replace parameter 'x' with 'value: i64'
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("calculate"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ParamReplaceOp::by_name("x", "value: i64"));
    
    apply_mutation(&mut buffer, &mutation).unwrap();
    
    println!("After replacing parameter 'x' with 'value: i64':");
    println!("{}", buffer.source());
}

fn demo_add_param() {
    println!("--- 2. AddParamOp: Add Parameter ---");
    
    let src = r#"
fn greet(name: String) -> String {
    format!("Hello, {}!", name)
}
"#;
    
    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/greet.rs"),
        src.to_string()
    ).unwrap();
    
    println!("Original:");
   println!("{}", buffer.source());
    
    // Add 'greeting: &str' parameter at the beginning
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("greet"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(AddParamOp::at_start("greeting: &str"));
    
    apply_mutation(&mut buffer, &mutation).unwrap();
    
    println!("After adding 'greeting: &str' at start:");
    println!("{}", buffer.source());
}

fn demo_remove_param() {
    println!("--- 3. RemoveParamOp: Remove Parameter ---");
    
    let src = r#"
fn process(input: String, debug: bool, verbose: bool) -> String {
    input.to_uppercase()
}
"#;
    
    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/process.rs"),
        src.to_string()
    ).unwrap();
    
    println!("Original:");
   println!("{}", buffer.source());
    
    // Remove 'debug' parameter
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("process"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(RemoveParamOp::by_name("debug"));
    
    apply_mutation(&mut buffer, &mutation).unwrap();
    
    println!("After removing 'debug' parameter:");
    println!("{}", buffer.source());
}

fn demo_body_replace() {
    println!("--- 4. BodyReplaceOp: Replace in Function Body ---");
    
    let src = r#"
fn compute() -> i32 {
    let result = 42;
    result * 2
}
"#;
    
    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/compute.rs"),
        src.to_string()
    ).unwrap();
    
    println!("Original:");
   println!("{}", buffer.source());
    
    // Replace 'result' with 'value' in function body
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("compute"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(BodyReplaceOp::new("result", "value"));
    
    apply_mutation(&mut buffer, &mutation).unwrap();
    
    println!("After replacing 'result' with 'value' in body:");
    println!("{}", buffer.source());
}
