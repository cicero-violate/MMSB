//! Example 5: Advanced Query Patterns
//!
//! Shows different ways to query the AST.

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use std::path::PathBuf;

fn main() {
    println!("=== Example 5: Query Patterns ===\n");

    let source = r#"
pub fn public_fn() {}
fn private_fn() {}

pub struct PublicStruct {}
struct PrivateStruct {}

pub trait PublicTrait {}

impl PublicTrait for PublicStruct {}
"#;

    let buffer = SourceBuffer::new(
        PathBuf::from("src/lib.rs"),
        source.to_string(),
    ).unwrap();

    println!("SOURCE:");
    println!("{}", buffer.source());

    // Pattern 1: Query by kind only
    println!("\n--- Pattern 1: All Functions ---");
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function));
    
    let results = execute_query(&buffer, &query);
    println!("Found {} functions", results.len());
    for item in results {
        if let syn::Item::Fn(f) = item {
            println!("  - {}", f.sig.ident);
        }
    }

    // Pattern 2: Query by name
    println!("\n--- Pattern 2: Specific Name ---");
    let query = QueryPlan::new()
        .with_predicate(NamePredicate::new("PublicStruct"));
    
    let results = execute_query(&buffer, &query);
    println!("Found {} items named 'PublicStruct'", results.len());

    // Pattern 3: Combined predicates
    println!("\n--- Pattern 3: Function + Name ---");
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("public_fn"));
    
    let results = execute_query(&buffer, &query);
    println!("Found {} items matching both predicates", results.len());

    // Pattern 4: All structs
    println!("\n--- Pattern 4: All Structs ---");
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Struct));
    
    let results = execute_query(&buffer, &query);
    println!("Found {} structs", results.len());
    for item in results {
        if let syn::Item::Struct(s) = item {
            println!("  - {}", s.ident);
        }
    }

    // Pattern 5: All traits
    println!("\n--- Pattern 5: All Traits ---");
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Trait));
    
    let results = execute_query(&buffer, &query);
    println!("Found {} traits", results.len());

    println!("\n💡 TIP: Custom predicates can be added for:");
    println!("  - Visibility checks (pub vs private)");
    println!("  - Generic parameters");
    println!("  - Attribute presence");
    println!("  - Complex AST patterns");
}
