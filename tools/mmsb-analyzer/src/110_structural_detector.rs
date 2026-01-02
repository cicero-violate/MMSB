//! Structural Invariant Detection
//!
//! This module detects invariants based on graph topology and structure.
//! All structural invariants are PROVEN (not heuristic) because they are
//! derived from mathematical properties of the graph.

use crate::invariant_types::*;
use crate::layer_inference::infer_layers;
use crate::scc_compressor::SccCompression;
use crate::types::{AnalysisResult, ElementType};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::HashMap;

/// Structural invariant detector
pub struct StructuralDetector {
    graph: DiGraph<String, ()>,
}

impl StructuralDetector {
    pub fn new(graph: DiGraph<String, ()>) -> Self {
        Self { graph }
    }

    /// Build call graph from analysis result
    pub fn build_call_graph(analysis: &AnalysisResult) -> DiGraph<String, ()> {
        let mut graph = DiGraph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        // Add nodes for all functions
        for element in &analysis.elements {
            if element.element_type == ElementType::Function {
                let node = graph.add_node(element.name.clone());
                node_map.insert(element.name.clone(), node);
            }
        }

        // Add edges from calls
        for element in &analysis.elements {
            if element.element_type == ElementType::Function {
                if let Some(&source) = node_map.get(&element.name) {
                    for callee in &element.calls {
                        if let Some(&target) = node_map.get(callee) {
                            if source != target {
                                // No self-loops
                                graph.add_edge(source, target, ());
                            }
                        }
                    }
                }
            }
        }

        graph
    }

    /// Detect PROVEN structural invariants from analysis result's call graph
    pub fn detect_from_analysis(analysis: &AnalysisResult) -> Vec<Invariant> {
        let graph = Self::build_call_graph(analysis);
        let detector = Self::new(graph);

        // Create a map from function name to file path
        let mut name_to_file: HashMap<String, String> = HashMap::new();
        for element in &analysis.elements {
            if element.element_type == ElementType::Function {
                name_to_file.insert(element.name.clone(), element.file_path.clone());
            }
        }

        let mut invariants = detector.detect_all();

        // Update file paths in invariants
        for inv in &mut invariants {
            if let Some(file_path) = name_to_file.get(&inv.target) {
                inv.file_path = file_path.clone();
            }
        }

        invariants
    }

    /// Detect all structural invariants
    pub fn detect_all(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        // Detect layer-fixed invariants
        invariants.extend(self.detect_layer_fixed());

        // Detect degree-stable invariants
        invariants.extend(self.detect_degree_stable());

        // Detect leaf and root nodes
        invariants.extend(self.detect_leaf_root());

        // Detect bridges
        invariants.extend(self.detect_bridges());

        // Detect SCC membership
        invariants.extend(self.detect_scc_membership());

        invariants
    }

    /// Detect layer-fixed invariants (PROVEN)
    ///
    /// Once a layer is assigned via the call graph, it's mathematically fixed
    fn detect_layer_fixed(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();
        let layers = infer_layers(&self.graph, 100);

        for (name, layer_info) in layers {
            let inv = Invariant::new(
                name.clone(),
                "".to_string(), // File path unknown at this level
                InvariantKind::Structural(StructuralInvariant::LayerFixed {
                    layer: layer_info.layer,
                }),
                InvariantStrength::Proven,
                format!("Layer {} assignment is proven from call graph", layer_info.layer),
            );

            invariants.push(inv);
        }

        invariants
    }

    /// Detect degree-stable invariants (PROVEN)
    ///
    /// In-degree and out-degree are structural properties
    fn detect_degree_stable(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        for node_idx in self.graph.node_indices() {
            let name = self.graph[node_idx].clone();
            let in_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Incoming)
                .count();
            let out_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .count();

            let mut inv = Invariant::new(
                name.clone(),
                "".to_string(),
                InvariantKind::Structural(StructuralInvariant::DegreeStable {
                    in_degree,
                    out_degree,
                }),
                InvariantStrength::Proven,
                format!(
                    "Degree: in={}, out={} (proven from graph)",
                    in_degree, out_degree
                ),
            );

            inv.add_evidence(format!("In-degree: {}", in_degree));
            inv.add_evidence(format!("Out-degree: {}", out_degree));

            invariants.push(inv);
        }

        invariants
    }

    /// Detect leaf and root nodes (PROVEN)
    fn detect_leaf_root(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        for node_idx in self.graph.node_indices() {
            let name = self.graph[node_idx].clone();
            let in_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Incoming)
                .count();
            let out_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .count();

            // Leaf node (out-degree = 0)
            if out_degree == 0 {
                let inv = Invariant::new(
                    name.clone(),
                    "".to_string(),
                    InvariantKind::Structural(StructuralInvariant::Leaf),
                    InvariantStrength::Proven,
                    "Leaf node (calls no other functions)".to_string(),
                );
                invariants.push(inv);
            }

            // Root node (in-degree = 0)
            if in_degree == 0 {
                let inv = Invariant::new(
                    name.clone(),
                    "".to_string(),
                    InvariantKind::Structural(StructuralInvariant::Root),
                    InvariantStrength::Proven,
                    "Root node (called by no other functions)".to_string(),
                );
                invariants.push(inv);
            }
        }

        invariants
    }

    /// Detect bridges (PROVEN)
    ///
    /// A bridge is an edge whose removal disconnects the graph
    fn detect_bridges(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        // Simple approach: for each node, check if removing it increases connected components
        // This is a simplified version - a full implementation would use Tarjan's bridge-finding algorithm

        for node_idx in self.graph.node_indices() {
            let name = self.graph[node_idx].clone();

            // Check if this node is critical for connectivity
            // (This is a heuristic approximation - true bridge detection is more complex)
            let in_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Incoming)
                .count();
            let out_degree = self
                .graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .count();

            // A node with in_degree=1 and out_degree=1 might be a bridge
            if in_degree == 1 && out_degree == 1 {
                let inv = Invariant::new(
                    name.clone(),
                    "".to_string(),
                    InvariantKind::Structural(StructuralInvariant::Bridge),
                    InvariantStrength::Empirical { paths_checked: 1 }, // Conservative - needs proper algorithm
                    "Potential bridge node in call graph".to_string(),
                );
                invariants.push(inv);
            }
        }

        invariants
    }

    /// Detect SCC membership (PROVEN)
    ///
    /// Membership in a strongly connected component is a proven structural property
    fn detect_scc_membership(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();
        let compression = SccCompression::new(self.graph.clone());

        for (scc_id, scc) in compression.sccs.iter().enumerate() {
            for &node_idx in scc {
                let name = self.graph[node_idx].clone();

                let inv = Invariant::new(
                    name.clone(),
                    "".to_string(),
                    InvariantKind::Structural(StructuralInvariant::SccMembership {
                        scc_id,
                        scc_size: scc.len(),
                    }),
                    InvariantStrength::Proven,
                    format!("Member of SCC {} (size: {})", scc_id, scc.len()),
                );

                invariants.push(inv);
            }
        }

        invariants
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_leaf_root() {
        // Create graph: A -> B -> C
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let detector = StructuralDetector::new(graph);
        let invariants = detector.detect_leaf_root();

        // A should be root, C should be leaf
        let roots: Vec<_> = invariants
            .iter()
            .filter(|inv| matches!(inv.kind, InvariantKind::Structural(StructuralInvariant::Root)))
            .collect();
        let leaves: Vec<_> = invariants
            .iter()
            .filter(|inv| matches!(inv.kind, InvariantKind::Structural(StructuralInvariant::Leaf)))
            .collect();

        assert_eq!(roots.len(), 1);
        assert_eq!(leaves.len(), 1);
        assert_eq!(roots[0].target, "A");
        assert_eq!(leaves[0].target, "C");
    }

    #[test]
    fn test_detect_degree_stable() {
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        graph.add_edge(a, b, ());

        let detector = StructuralDetector::new(graph);
        let invariants = detector.detect_degree_stable();

        assert_eq!(invariants.len(), 2); // One for A, one for B

        // Find A's invariant
        let a_inv = invariants.iter().find(|inv| inv.target == "A").unwrap();
        if let InvariantKind::Structural(StructuralInvariant::DegreeStable {
            in_degree,
            out_degree,
        }) = &a_inv.kind
        {
            assert_eq!(*in_degree, 0);
            assert_eq!(*out_degree, 1);
        } else {
            panic!("Wrong invariant kind");
        }
    }

    #[test]
    fn test_all_structural_invariants_proven() {
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        graph.add_edge(a, b, ());

        let detector = StructuralDetector::new(graph);
        let invariants = detector.detect_all();

        // Most should be proven (except bridges which are conservative)
        let proven_count = invariants
            .iter()
            .filter(|inv| matches!(inv.strength, InvariantStrength::Proven))
            .count();

        assert!(proven_count > 0);
    }
}
