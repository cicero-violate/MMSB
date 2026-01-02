//! SCC (Strongly Connected Components) Compression
//!
//! This module compresses strongly connected components in a directed graph
//! to prevent path explosion during invariant detection. Each SCC is compressed
//! into a single node, guaranteeing the resulting graph is a DAG.

use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// A compressed graph where each SCC is represented as a single node
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SccCompression {
    /// Original graph
    pub original_graph: DiGraph<String, ()>,

    /// Compressed graph (guaranteed to be a DAG)
    pub compressed_graph: DiGraph<Vec<String>, ()>,

    /// Mapping from original node index to SCC ID
    pub node_to_scc: HashMap<NodeIndex, usize>,

    /// Mapping from SCC ID to compressed node index
    pub scc_to_node: HashMap<usize, NodeIndex>,

    /// All SCCs detected
    pub sccs: Vec<Vec<NodeIndex>>,
}

#[allow(dead_code)]
impl SccCompression {
    /// Create a new SCC compression from a call graph
    ///
    /// # Arguments
    /// * `graph` - The directed graph to compress
    ///
    /// # Returns
    /// A compressed representation where each SCC is a single node
    pub fn new(graph: DiGraph<String, ()>) -> Self {
        // Step 1: Detect SCCs using Tarjan's algorithm
        let sccs = tarjan_scc(&graph);

        // Step 2: Build mapping from node to SCC ID
        let mut node_to_scc = HashMap::new();
        for (scc_id, scc) in sccs.iter().enumerate() {
            for &node in scc {
                node_to_scc.insert(node, scc_id);
            }
        }

        // Step 3: Build compressed graph
        let mut compressed_graph = DiGraph::new();
        let mut scc_to_node = HashMap::new();

        // Add one node per SCC
        for (scc_id, scc) in sccs.iter().enumerate() {
            let members: Vec<String> = scc
                .iter()
                .map(|&node_idx| graph[node_idx].clone())
                .collect();
            let compressed_node = compressed_graph.add_node(members);
            scc_to_node.insert(scc_id, compressed_node);
        }

        // Add edges between SCCs (but not within SCCs)
        for edge in graph.edge_references() {
            let source_scc = node_to_scc[&edge.source()];
            let target_scc = node_to_scc[&edge.target()];

            // Only add edge if it goes between different SCCs
            if source_scc != target_scc {
                let source_node = scc_to_node[&source_scc];
                let target_node = scc_to_node[&target_scc];

                // Check if edge already exists to avoid duplicates
                if !compressed_graph.contains_edge(source_node, target_node) {
                    compressed_graph.add_edge(source_node, target_node, ());
                }
            }
        }

        Self {
            original_graph: graph,
            compressed_graph,
            node_to_scc,
            scc_to_node,
            sccs,
        }
    }

    /// Check if the compressed graph is a DAG
    ///
    /// This should always return true after SCC compression
    pub fn is_dag(&self) -> bool {
        petgraph::algo::is_cyclic_directed(&self.compressed_graph) == false
    }

    /// Get the SCC containing a node name
    pub fn get_scc_for_node(&self, node_name: &str) -> Option<usize> {
        for (node_idx, name) in self.original_graph.node_weights().enumerate() {
            if name == node_name {
                let node_index = NodeIndex::new(node_idx);
                return self.node_to_scc.get(&node_index).copied();
            }
        }
        None
    }

    /// Get all members of an SCC
    pub fn get_scc_members(&self, scc_id: usize) -> Vec<String> {
        if let Some(scc) = self.sccs.get(scc_id) {
            scc.iter()
                .map(|&node_idx| self.original_graph[node_idx].clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Count trivial SCCs (size 1)
    pub fn count_trivial_sccs(&self) -> usize {
        self.sccs.iter().filter(|scc| scc.len() == 1).count()
    }

    /// Count non-trivial SCCs (size > 1, i.e., cycles)
    pub fn count_nontrivial_sccs(&self) -> usize {
        self.sccs.iter().filter(|scc| scc.len() > 1).count()
    }

    /// Get statistics about the compression
    pub fn stats(&self) -> SccStats {
        SccStats {
            original_node_count: self.original_graph.node_count(),
            compressed_node_count: self.compressed_graph.node_count(),
            trivial_sccs: self.count_trivial_sccs(),
            nontrivial_sccs: self.count_nontrivial_sccs(),
            is_dag: self.is_dag(),
        }
    }
}

/// Statistics about SCC compression
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SccStats {
    pub original_node_count: usize,
    pub compressed_node_count: usize,
    pub trivial_sccs: usize,
    pub nontrivial_sccs: usize,
    pub is_dag: bool,
}

impl std::fmt::Display for SccStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SCC Compression: {} nodes → {} SCCs ({} trivial, {} cycles) - DAG: {}",
            self.original_node_count,
            self.compressed_node_count,
            self.trivial_sccs,
            self.nontrivial_sccs,
            self.is_dag
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scc_compression_dag() {
        // Create a simple DAG: A -> B -> C
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let compression = SccCompression::new(graph);

        assert!(compression.is_dag());
        assert_eq!(compression.count_trivial_sccs(), 3);
        assert_eq!(compression.count_nontrivial_sccs(), 0);
    }

    #[test]
    fn test_scc_compression_cycle() {
        // Create a graph with a cycle: A -> B -> C -> A
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, a, ());

        let compression = SccCompression::new(graph);

        assert!(compression.is_dag());
        assert_eq!(compression.count_nontrivial_sccs(), 1);
        assert_eq!(compression.compressed_graph.node_count(), 1);
    }

    #[test]
    fn test_scc_compression_mixed() {
        // Create a graph with a cycle and a leaf: A -> B -> C -> B, A -> D
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        let d = graph.add_node("D".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, b, ()); // Cycle: B <-> C
        graph.add_edge(a, d, ());

        let compression = SccCompression::new(graph);

        assert!(compression.is_dag());
        assert_eq!(compression.count_nontrivial_sccs(), 1); // B-C cycle
        assert_eq!(compression.count_trivial_sccs(), 2); // A and D
        assert_eq!(compression.compressed_graph.node_count(), 3);
    }
}
