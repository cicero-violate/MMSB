//! Snapshot Export
//!
//! Serializes DependencyGraph + PropagationStats with deterministic hash.

use mmsb_core::adaptive::PropagationStats;
use mmsb_core::dag::DependencyGraph;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};

/// Indexed snapshot ready for MMSB core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedSnapshot {
    pub dag: DependencyGraph,
    pub stats: PropagationStats,
    pub snapshot_hash: String,
}

impl IndexedSnapshot {
    /// Create snapshot with computed hash
    pub fn new(dag: DependencyGraph, stats: PropagationStats) -> Self {
        let snapshot_hash = compute_snapshot_hash(&dag);
        
        Self {
            dag,
            stats,
            snapshot_hash,
        }
    }
    
    /// Serialize to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
    
    /// Deserialize from JSON
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }
}

/// Compute deterministic hash of dependency graph
///
/// **CRITICAL:** Must be deterministic for same graph structure
fn compute_snapshot_hash(dag: &DependencyGraph) -> String {
    let mut hasher = Sha256::new();
    
    // Sort edges for determinism
    let mut edges = dag.edges();
    edges.sort_by_key(|(from, to, edge_type)| {
        (from.0, to.0, *edge_type as u8)
    });
    
    // Hash sorted edges
    for (from, to, edge_type) in edges {
        hasher.update(&from.0.to_le_bytes());
        hasher.update(&to.0.to_le_bytes());
        hasher.update(&[edge_type as u8]);
    }
    
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmsb_core::dag::{build_dependency_graph, EdgeType, StructuralOp};
    use mmsb_core::types::PageID;
    
    #[test]
    fn test_snapshot_hash_deterministic() {
        let ops = vec![
            StructuralOp::AddEdge {
                from: PageID(1),
                to: PageID(2),
                edge_type: EdgeType::Data,
            },
        ];
        let dag1 = build_dependency_graph(&ops);
        let dag2 = build_dependency_graph(&ops);
        
        let hash1 = compute_snapshot_hash(&dag1);
        let hash2 = compute_snapshot_hash(&dag2);
        
        assert_eq!(hash1, hash2);
    }
    
    #[test]
    fn test_snapshot_serialization() {
        let dag = DependencyGraph::new();
        let stats = PropagationStats::new();
        let snapshot = IndexedSnapshot::new(dag, stats);
        
        let json = snapshot.to_json().unwrap();
        let deserialized = IndexedSnapshot::from_json(&json).unwrap();
        
        assert_eq!(snapshot.snapshot_hash, deserialized.snapshot_hash);
    }
}
