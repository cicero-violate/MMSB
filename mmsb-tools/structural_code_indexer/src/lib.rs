pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

/// Structural Code Indexer
///
/// **Read-only** ingestion layer for MMSB.
/// Converts source code into DependencyGraph + PropagationStats.
///
/// **CRITICAL CONSTRAINTS:**
/// - Read-only: Never modifies source files
/// - Deterministic: Same repo → same snapshot hash
/// - No judgment, no commits, no proposals

pub mod fs;
pub mod extract;
pub mod index;
pub mod graph;
pub mod stats;
pub mod export;

use std::path::Path;
pub use export::IndexedSnapshot;

/// Index a repository and produce a snapshot
///
/// **Public API:** This is the primary entry point.
///
/// **Guarantees:**
/// - Read-only (no file mutations)
/// - Deterministic (same repo → same hash)
/// - Complete (all observed references included)
///
/// **No side effects.**
pub fn index_repository(root: &Path) -> IndexedSnapshot {
    // Step 1: Read source files
    let source_files = fs::read_source_files(root);
    
    // Step 2: Extract references
    let mut refs = Vec::new();
    for file in &source_files {
        let extracted = extract::extract_rust_refs(file.path.clone(), &file.content);
        refs.extend(extracted);
    }
    
    // Step 3: Build indices
    let mut page_index = index::PageIndex::new();
    let mut symbol_index = index::SymbolIndex::new();
    
    // Step 4: Build dependency graph
    let dag = graph::build_dependency_graph(&refs, &mut page_index, &mut symbol_index);
    
    // Step 5: Compute propagation statistics
    let stats = stats::compute_propagation_stats(&dag);
    
    // Step 6: Create snapshot
    export::IndexedSnapshot::new(dag, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_empty_directory() {
        use std::fs;
        use tempfile::TempDir;
        
        let temp = TempDir::new().unwrap();
        let snapshot = index_repository(temp.path());
        
        // Empty repo should have deterministic hash
        assert!(!snapshot.snapshot_hash.is_empty());
    }
}
