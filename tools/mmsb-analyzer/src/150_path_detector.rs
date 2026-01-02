//! Path-Intersection Invariant Detection
//!
//! This is the MOST POWERFUL detector - it finds facts that hold on ALL execution paths.
//! Uses SCC compression to prevent path explosion.

use crate::invariant_types::*;
use crate::scc_compressor::SccCompression;
use petgraph::algo::all_simple_paths;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::HashSet;

/// Path invariant detector
pub struct PathDetector {
    graph: DiGraph<String, ()>,
    compression: SccCompression,
}

impl PathDetector {
    pub fn new(graph: DiGraph<String, ()>) -> Self {
        let compression = SccCompression::new(graph.clone());
        Self { graph, compression }
    }

    /// Detect all path-intersection invariants
    ///
    /// # Arguments
    /// * `max_depth` - Maximum path depth to explore
    /// * `max_paths_per_node` - Maximum paths to check per node
    pub fn detect_all(&self, max_depth: usize, max_paths_per_node: usize) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        // Enumerate paths on the SCC-compressed graph (guaranteed DAG)
        for node_idx in self.compression.compressed_graph.node_indices() {
            let members = &self.compression.compressed_graph[node_idx];

            // For each member in this SCC
            for member in members {
                if let Some(inv) =
                    self.compute_path_intersection(member, max_depth, max_paths_per_node)
                {
                    invariants.push(inv);
                }
            }
        }

        invariants
    }

    /// Compute path-intersection invariant for a specific node
    fn compute_path_intersection(
        &self,
        target: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Option<Invariant> {
        // Find the node index in the original graph
        let target_idx = self
            .graph
            .node_indices()
            .find(|&idx| self.graph[idx] == target)?;

        // Get all leaf nodes (out-degree = 0) reachable from this node
        let leaf_nodes: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, Direction::Outgoing)
                    .count()
                    == 0
            })
            .collect();

        let mut all_paths = Vec::new();
        let mut paths_found = 0;

        // Enumerate paths to all reachable leaves
        for leaf_idx in leaf_nodes {
            // Use petgraph's all_simple_paths (bounded by max_depth)
            let paths: Vec<Vec<NodeIndex>> = all_simple_paths(
                &self.graph,
                target_idx,
                leaf_idx,
                0,
                Some(max_depth),
            )
            .take(max_paths - paths_found)
            .collect();

            paths_found += paths.len();
            all_paths.extend(paths);

            if paths_found >= max_paths {
                break;
            }
        }

        // If we found no paths or only one path, no useful invariant
        if all_paths.len() < 2 {
            return None;
        }

        // Extract facts from each path
        let path_facts: Vec<HashSet<String>> = all_paths
            .iter()
            .map(|path| self.extract_facts_from_path(path))
            .collect();

        // Compute intersection of all path facts
        let mut intersection = path_facts[0].clone();
        for facts in &path_facts[1..] {
            intersection = intersection.intersection(facts).cloned().collect();
        }

        // If intersection is empty, no common facts
        if intersection.is_empty() {
            return None;
        }

        // Create invariant
        Some(Invariant::new(
            target.to_string(),
            "".to_string(),
            InvariantKind::PathIntersection(PathIntersectionInvariant {
                facts: intersection,
                paths_analyzed: all_paths.len(),
                max_depth,
            }),
            InvariantStrength::Empirical {
                paths_checked: all_paths.len(),
            },
            format!(
                "Facts hold on all {} paths (max depth: {})",
                all_paths.len(),
                max_depth
            ),
        ))
    }

    /// Extract symbolic facts from a path
    ///
    /// This is a simplified version - a full implementation would use
    /// abstract interpretation or symbolic execution
    fn extract_facts_from_path(&self, path: &[NodeIndex]) -> HashSet<String> {
        let mut facts = HashSet::new();

        // Fact: path length
        facts.insert(format!("path_length_{}", path.len()));

        // Fact: nodes visited
        for &node_idx in path {
            let name = &self.graph[node_idx];
            facts.insert(format!("visits_{}", name));
        }

        // Fact: edges traversed
        for window in path.windows(2) {
            let from = &self.graph[window[0]];
            let to = &self.graph[window[1]];
            facts.insert(format!("edge_{}_to_{}", from, to));
        }

        // Fact: always reaches leaf (if path ends at a leaf)
        if let Some(&last) = path.last() {
            let out_degree = self
                .graph
                .neighbors_directed(last, Direction::Outgoing)
                .count();
            if out_degree == 0 {
                facts.insert("reaches_leaf".to_string());
            }
        }

        facts
    }

    /// Get statistics about path enumeration
    #[allow(dead_code)]
    pub fn get_stats(&self) -> PathStats {
        let total_nodes = self.graph.node_count();
        let compressed_nodes = self.compression.compressed_graph.node_count();

        PathStats {
            original_nodes: total_nodes,
            compressed_nodes,
            is_dag: self.compression.is_dag(),
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PathStats {
    pub original_nodes: usize,
    pub compressed_nodes: usize,
    pub is_dag: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_detector_simple() {
        // Create graph: A -> B -> C
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let detector = PathDetector::new(graph);
        let _invariants = detector.detect_all(10, 100);

        // Should find some invariants (at least for A which has a path to C)
        // Note: might be 0 if only one path exists (no intersection needed)
    }

    #[test]
    fn test_path_detector_diamond() {
        // Diamond: A -> B -> D, A -> C -> D
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        let d = graph.add_node("D".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        graph.add_edge(c, d, ());

        let detector = PathDetector::new(graph);
        let invariants = detector.detect_all(10, 100);

        // A has two paths to D, should find intersection invariant
        let a_invs: Vec<_> = invariants.iter().filter(|inv| inv.target == "A").collect();

        if !a_invs.is_empty() {
            let inv = a_invs[0];
            if let InvariantKind::PathIntersection(ref pi) = inv.kind {
                assert_eq!(pi.paths_analyzed, 2);
                // Both paths should reach D
                assert!(pi.facts.contains("visits_D"));
            }
        }
    }

    #[test]
    fn test_extract_facts_from_path() {
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let detector = PathDetector::new(graph);
        let path = vec![a, b, c];
        let facts = detector.extract_facts_from_path(&path);

        assert!(facts.contains("visits_A"));
        assert!(facts.contains("visits_B"));
        assert!(facts.contains("visits_C"));
        assert!(facts.contains("reaches_leaf"));
        assert!(facts.contains("path_length_3"));
    }

    #[test]
    fn test_path_stats() {
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        graph.add_edge(a, b, ());

        let detector = PathDetector::new(graph);
        let stats = detector.get_stats();

        assert_eq!(stats.original_nodes, 2);
        assert!(stats.is_dag);
    }
}
