//! Fixpoint Solver for Symbolic Abstraction Propagation
//!
//! This module propagates symbolic abstractions through the call graph
//! until a fixpoint is reached. Abstractions that remain stable at the
//! fixpoint are candidate invariants.

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use std::collections::{HashMap, HashSet};

/// Symbolic abstraction of program state
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub struct SymbolicAbstraction {
    /// Type signature
    pub type_sig: Option<String>,

    /// Effect signature (I/O, mutation, etc.)
    pub effects: HashSet<String>,

    /// Layer assignment
    pub layer: Option<usize>,

    /// Ownership/visibility
    pub visibility: Option<String>,

    /// Additional properties
    pub properties: HashSet<String>,
}

impl SymbolicAbstraction {
    pub fn new() -> Self {
        Self {
            type_sig: None,
            effects: HashSet::new(),
            layer: None,
            visibility: None,
            properties: HashSet::new(),
        }
    }

    /// Check if two abstractions are approximately equal
    pub fn approx_eq(&self, other: &Self) -> bool {
        self.type_sig == other.type_sig
            && self.effects == other.effects
            && self.layer == other.layer
            && self.visibility == other.visibility
            && self.properties == other.properties
    }

    /// Merge two abstractions (used during propagation)
    pub fn merge(&mut self, other: &Self) {
        // Type signature: take most specific (if both exist, keep current)
        if self.type_sig.is_none() && other.type_sig.is_some() {
            self.type_sig = other.type_sig.clone();
        }

        // Effects: union
        self.effects.extend(other.effects.clone());

        // Layer: take maximum
        match (self.layer, other.layer) {
            (Some(l1), Some(l2)) => self.layer = Some(l1.max(l2)),
            (None, Some(l)) => self.layer = Some(l),
            _ => {}
        }

        // Visibility: take most restrictive
        if self.visibility.is_none() && other.visibility.is_some() {
            self.visibility = other.visibility.clone();
        }

        // Properties: union
        self.properties.extend(other.properties.clone());
    }
}

impl Default for SymbolicAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of fixpoint computation
#[derive(Debug)]
#[allow(dead_code)]
pub struct FixpointResult {
    /// Final abstraction for each node
    pub abstractions: HashMap<String, SymbolicAbstraction>,

    /// Number of iterations to convergence
    pub iterations: usize,

    /// Whether fixpoint was reached
    pub converged: bool,

    /// Nodes whose abstractions were stable
    pub stable_nodes: Vec<String>,
}

/// Propagate symbolic abstractions until fixpoint
///
/// # Arguments
/// * `graph` - Call graph
/// * `initial` - Initial abstractions for each node
/// * `max_iterations` - Maximum iterations before giving up
///
/// # Returns
/// FixpointResult containing final abstractions and metadata
#[allow(dead_code)]
pub fn propagate_to_fixpoint(
    graph: &DiGraph<String, ()>,
    initial: HashMap<String, SymbolicAbstraction>,
    max_iterations: usize,
) -> FixpointResult {
    // Build mapping from node name to index
    let mut name_to_idx: HashMap<String, NodeIndex> = HashMap::new();
    for idx in graph.node_indices() {
        name_to_idx.insert(graph[idx].clone(), idx);
    }

    // Initialize abstractions
    let mut current: HashMap<NodeIndex, SymbolicAbstraction> = HashMap::new();
    for (name, abstraction) in &initial {
        if let Some(&idx) = name_to_idx.get(name) {
            current.insert(idx, abstraction.clone());
        }
    }

    // Fixpoint iteration
    let mut iteration = 0;
    let mut converged = false;

    while iteration < max_iterations {
        iteration += 1;
        let mut next = current.clone();
        let mut changed = false;

        // Propagate from callees to callers
        for node_idx in graph.node_indices() {
            let mut new_abstraction = current
                .get(&node_idx)
                .cloned()
                .unwrap_or_else(SymbolicAbstraction::new);

            // Get all callees (outgoing edges)
            for callee_idx in graph.neighbors_directed(node_idx, Direction::Outgoing) {
                if let Some(callee_abs) = current.get(&callee_idx) {
                    new_abstraction.merge(callee_abs);
                }
            }

            // Check if changed
            if let Some(old) = current.get(&node_idx) {
                if !old.approx_eq(&new_abstraction) {
                    changed = true;
                }
            } else {
                changed = true;
            }

            next.insert(node_idx, new_abstraction);
        }

        current = next;

        if !changed {
            converged = true;
            break;
        }
    }

    // Convert back to name-based mapping
    let mut abstractions = HashMap::new();
    let mut stable_nodes = Vec::new();

    for (idx, abstraction) in &current {
        let name = graph[*idx].clone();

        // Check if this abstraction matches initial (stable)
        if let Some(initial_abs) = initial.get(&name) {
            if initial_abs.approx_eq(abstraction) {
                stable_nodes.push(name.clone());
            }
        }

        abstractions.insert(name, abstraction.clone());
    }

    FixpointResult {
        abstractions,
        iterations: iteration,
        converged,
        stable_nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_abstraction_merge() {
        let mut abs1 = SymbolicAbstraction::new();
        abs1.type_sig = Some("String".to_string());
        abs1.effects.insert("IO".to_string());
        abs1.layer = Some(1);

        let mut abs2 = SymbolicAbstraction::new();
        abs2.effects.insert("Mutation".to_string());
        abs2.layer = Some(2);

        abs1.merge(&abs2);

        assert_eq!(abs1.type_sig, Some("String".to_string()));
        assert!(abs1.effects.contains("IO"));
        assert!(abs1.effects.contains("Mutation"));
        assert_eq!(abs1.layer, Some(2)); // max
    }

    #[test]
    fn test_fixpoint_simple() {
        // Graph: A -> B -> C
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());

        // Initial: C has effect "Pure"
        let mut initial = HashMap::new();
        let mut c_abs = SymbolicAbstraction::new();
        c_abs.effects.insert("Pure".to_string());
        initial.insert("C".to_string(), c_abs);

        initial.insert("A".to_string(), SymbolicAbstraction::new());
        initial.insert("B".to_string(), SymbolicAbstraction::new());

        let result = propagate_to_fixpoint(&graph, initial, 100);

        assert!(result.converged);
        // A and B should propagate "Pure" from C
        assert!(result.abstractions["A"].effects.contains("Pure"));
        assert!(result.abstractions["B"].effects.contains("Pure"));
    }

    #[test]
    fn test_fixpoint_convergence() {
        // Simple chain should converge quickly
        let mut graph = DiGraph::new();
        let a = graph.add_node("A".to_string());
        let b = graph.add_node("B".to_string());
        graph.add_edge(a, b, ());

        let mut initial = HashMap::new();
        initial.insert("A".to_string(), SymbolicAbstraction::new());
        initial.insert("B".to_string(), SymbolicAbstraction::new());

        let result = propagate_to_fixpoint(&graph, initial, 100);

        assert!(result.converged);
        assert!(result.iterations < 10);
    }
}
