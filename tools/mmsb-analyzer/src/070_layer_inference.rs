//! Layer Inference from Call Graph
//!
//! This module infers layer assignments from the call graph structure,
//! NOT from filename prefixes. This makes the system self-correcting
//! and prevents silent drift.
//!
//! Algorithm:
//! 1. Assign layer 0 to all leaf nodes (out-degree = 0)
//! 2. Repeat until fixpoint:
//!    - For each unassigned node:
//!      - If all callees have layers assigned:
//!        - layer(node) = max(layer(callee)) + 1
//! 3. Remaining nodes get max + 1

use crate::invariant_types::LayerInfo;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashMap;

/// Infer layers from call graph structure
///
/// # Arguments
/// * `graph` - Call graph (edge from caller to callee)
/// * `max_iterations` - Maximum fixpoint iterations (default: 100)
///
/// # Returns
/// HashMap mapping node name to layer information
pub fn infer_layers(graph: &DiGraph<String, ()>, max_iterations: usize) -> HashMap<String, LayerInfo> {
    let mut layers: HashMap<NodeIndex, usize> = HashMap::new();
    let mut result: HashMap<String, LayerInfo> = HashMap::new();

    // Step 1: Assign layer 0 to all leaf nodes (out-degree = 0)
    for node_idx in graph.node_indices() {
        let out_degree = graph.neighbors_directed(node_idx, Direction::Outgoing).count();
        if out_degree == 0 {
            layers.insert(node_idx, 0);
        }
    }

    // Step 2: Fixpoint iteration
    let mut changed = true;
    let mut iteration = 0;

    while changed && iteration < max_iterations {
        changed = false;
        iteration += 1;

        for node_idx in graph.node_indices() {
            // Skip if already assigned
            if layers.contains_key(&node_idx) {
                continue;
            }

            // Get all callees (outgoing edges)
            let callees: Vec<NodeIndex> = graph
                .neighbors_directed(node_idx, Direction::Outgoing)
                .collect();

            // Check if all callees have layer assignments
            let all_assigned = callees.iter().all(|callee| layers.contains_key(callee));

            if all_assigned && !callees.is_empty() {
                // Compute layer = max(callee layers) + 1
                let max_callee_layer = callees
                    .iter()
                    .filter_map(|callee| layers.get(callee))
                    .max()
                    .copied()
                    .unwrap_or(0);

                layers.insert(node_idx, max_callee_layer + 1);
                changed = true;
            }
        }
    }

    // Step 3: Assign remaining nodes (those in cycles or unreachable)
    let max_layer = layers.values().max().copied().unwrap_or(0);
    for node_idx in graph.node_indices() {
        if !layers.contains_key(&node_idx) {
            layers.insert(node_idx, max_layer + 1);
        }
    }

    // Step 4: Build LayerInfo for each node
    for (node_idx, &layer) in &layers {
        let name = graph[*node_idx].clone();

        // Get dependencies (callees)
        let dependencies: Vec<String> = graph
            .neighbors_directed(*node_idx, Direction::Outgoing)
            .map(|callee_idx| graph[callee_idx].clone())
            .collect();

        // Get max dependency layer
        let max_dependency_layer = if dependencies.is_empty() {
            None
        } else {
            dependencies
                .iter()
                .filter_map(|dep_name| {
                    graph
                        .node_indices()
                        .find(|idx| &graph[*idx] == dep_name)
                        .and_then(|idx| layers.get(&idx))
                        .copied()
                })
                .max()
        };

        result.insert(
            name.clone(),
            LayerInfo {
                name,
                layer,
                dependencies,
                max_dependency_layer,
            },
        );
    }

    result
}

/// Detect layer violations in the call graph
///
/// A violation occurs when a lower-layer function calls a higher-layer function
///
/// # Arguments
/// * `layer_assignments` - Layer information for each node
/// * `graph` - The call graph
///
/// # Returns
/// Vec of (caller, callee, caller_layer, callee_layer) tuples
pub fn detect_layer_violations(
    layer_assignments: &HashMap<String, LayerInfo>,
    graph: &DiGraph<String, ()>,
) -> Vec<(String, String, usize, usize)> {
    let mut violations = Vec::new();

    for edge in graph.edge_references() {
        let caller = &graph[edge.source()];
        let callee = &graph[edge.target()];

        if let (Some(caller_info), Some(callee_info)) =
            (layer_assignments.get(caller), layer_assignments.get(callee))
        {
            // Violation: caller layer <= callee layer (should be strictly greater)
            // Actually, caller should be > callee (higher layers call lower layers)
            if caller_info.layer <= callee_info.layer {
                violations.push((
                    caller.clone(),
                    callee.clone(),
                    caller_info.layer,
                    callee_info.layer,
                ));
            }
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_inference_simple_dag() {
        // Create a simple DAG: A -> B -> C
        // Expected layers: C=0, B=1, A=2
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let layers = infer_layers(&graph, 100);

        assert_eq!(layers.get("C").unwrap().layer, 0); // Leaf
        assert_eq!(layers.get("B").unwrap().layer, 1);
        assert_eq!(layers.get("A").unwrap().layer, 2);
    }

    #[test]
    fn test_layer_inference_diamond() {
        // Diamond: A -> B -> D, A -> C -> D
        // Expected layers: D=0, B=1, C=1, A=2
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        let d = graph.add_node("D".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());
        graph.add_edge(b, d, ());
        graph.add_edge(c, d, ());

        let layers = infer_layers(&graph, 100);

        assert_eq!(layers.get("D").unwrap().layer, 0);
        assert_eq!(layers.get("B").unwrap().layer, 1);
        assert_eq!(layers.get("C").unwrap().layer, 1);
        assert_eq!(layers.get("A").unwrap().layer, 2);
    }

    #[test]
    fn test_detect_layer_violations_none() {
        // Proper layering: A(2) -> B(1) -> C(0)
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        let layers = infer_layers(&graph, 100);
        let violations = detect_layer_violations(&layers, &graph);

        assert_eq!(violations.len(), 0);
    }
}
