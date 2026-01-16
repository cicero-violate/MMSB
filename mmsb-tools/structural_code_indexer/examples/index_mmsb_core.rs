//! Example: Index MMSB Core Repository
//!
//! Demonstrates using the structural code indexer to analyze the mmsb-core codebase.

use structural_code_indexer::index_repository;
use std::path::Path;

fn main() {
    println!("=== MMSB Structural Code Indexer Example ===\n");
    
    // Index the mmsb-core repository
    let mmsb_core_path = Path::new("../../mmsb-core");
    
    println!("Indexing repository: {}", mmsb_core_path.display());
    println!("Reading source files...");
    
    let snapshot = index_repository(mmsb_core_path);
    
    println!("\n=== Indexing Results ===");
    println!("Snapshot hash: {}", snapshot.snapshot_hash);
    
    // Count edges
    let edges = snapshot.dag.edges();
    println!("\nDependency Graph:");
    println!("  Total edges: {}", edges.len());
    
    // Count edge types
    let mut data_edges = 0;
    let mut control_edges = 0;
    let mut gpu_edges = 0;
    let mut compiler_edges = 0;
    
    for (_, _, edge_type) in &edges {
        match edge_type {
            mmsb_core::dag::EdgeType::Data => data_edges += 1,
            mmsb_core::dag::EdgeType::Control => control_edges += 1,
            mmsb_core::dag::EdgeType::Gpu => gpu_edges += 1,
            mmsb_core::dag::EdgeType::Compiler => compiler_edges += 1,
        }
    }
    
    println!("  Data edges: {}", data_edges);
    println!("  Control edges: {}", control_edges);
    println!("  GPU edges: {}", gpu_edges);
    println!("  Compiler edges: {}", compiler_edges);
    
    // Propagation stats
    println!("\nPropagation Statistics:");
    println!("  Total propagations: {}", snapshot.stats.total_propagations);
    println!("  Pages with fanout: {}", snapshot.stats.fanout_per_page.len());
    
    if let Some(median) = snapshot.stats.median_fanout() {
        println!("  Median fanout: {}", median);
    }
    
    // High fanout pages
    let high_fanout = snapshot.stats.high_fanout_pages(5);
    if !high_fanout.is_empty() {
        println!("\nHigh-fanout pages (>5):");
        for (page, fanout) in high_fanout.iter().take(10) {
            println!("  PageID({}) → {} dependencies", page.0, fanout);
        }
    }
    
    // Zero fanout pages
    let zero_fanout = snapshot.stats.zero_fanout_pages();
    if !zero_fanout.is_empty() {
        println!("\nPages with zero fanout: {}", zero_fanout.len());
    }
    
    // Serialize to JSON
    println!("\n=== Serialization ===");
    match snapshot.to_json() {
        Ok(json) => {
            let size = json.len();
            println!("JSON size: {} bytes", size);
            println!("First 200 chars:");
            println!("{}", &json[..200.min(size)]);
        }
        Err(e) => {
            eprintln!("Failed to serialize: {}", e);
        }
    }
    
    println!("\n=== Example Complete ===");
}
