//! Example: Index a Simple Repository
//!
//! Demonstrates basic usage on a small example.

use structural_code_indexer::index_repository;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

fn main() {
    println!("=== Simple Repository Indexing Example ===\n");
    
    // Create a temporary repository
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let repo_path = temp_dir.path();
    
    println!("Creating example repository at: {}", repo_path.display());
    
    // Create example source files
    create_example_repo(repo_path);
    
    // Index the repository
    println!("\nIndexing repository...");
    let snapshot = index_repository(repo_path);
    
    // Display results
    println!("\n=== Results ===");
    println!("Snapshot hash: {}", snapshot.snapshot_hash);
    
    let edges = snapshot.dag.edges();
    println!("\nDependency Graph:");
    println!("  Total edges: {}", edges.len());
    
    if !edges.is_empty() {
        println!("\n  Edges:");
        for (from, to, edge_type) in edges.iter().take(10) {
            println!("    PageID({}) → PageID({}) [{:?}]", from.0, to.0, edge_type);
        }
        if edges.len() > 10 {
            println!("    ... and {} more", edges.len() - 10);
        }
    }
    
    println!("\nPropagation Statistics:");
    println!("  Pages with fanout: {}", snapshot.stats.fanout_per_page.len());
    
    for (page, fanout) in snapshot.stats.fanout_per_page.iter().take(5) {
        println!("    PageID({}) has fanout: {}", page.0, fanout);
    }
    
    println!("\n=== Example Complete ===");
}

fn create_example_repo(repo_path: &Path) {
    // Create directory structure
    fs::create_dir_all(repo_path.join("src")).expect("Failed to create src dir");
    
    // lib.rs
    fs::write(
        repo_path.join("src/lib.rs"),
        r#"
//! Example library

pub mod foo;
pub mod bar;

pub use foo::Foo;
pub use bar::Bar;

pub fn main_func() {
    let f = Foo::new();
    let b = Bar::new();
    f.process(&b);
}
"#,
    )
    .expect("Failed to write lib.rs");
    
    // foo.rs
    fs::write(
        repo_path.join("src/foo.rs"),
        r#"
use crate::bar::Bar;
use std::collections::HashMap;

pub struct Foo {
    data: HashMap<String, i32>,
}

impl Foo {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    pub fn process(&self, bar: &Bar) {
        bar.execute();
    }
}
"#,
    )
    .expect("Failed to write foo.rs");
    
    // bar.rs
    fs::write(
        repo_path.join("src/bar.rs"),
        r#"
use std::sync::Arc;

pub struct Bar {
    state: Arc<String>,
}

impl Bar {
    pub fn new() -> Self {
        Self {
            state: Arc::new("initialized".to_string()),
        }
    }
    
    pub fn execute(&self) {
        println!("Executing with state: {}", self.state);
    }
}
"#,
    )
    .expect("Failed to write bar.rs");
    
    println!("  ✓ Created src/lib.rs");
    println!("  ✓ Created src/foo.rs");
    println!("  ✓ Created src/bar.rs");
}
