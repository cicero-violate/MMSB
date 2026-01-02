//! Invariant Detection Integration
//!
//! This module orchestrates all invariant detectors and produces a unified result.

use crate::invariant_types::*;
use crate::layer_inference::{detect_layer_violations, infer_layers};
use crate::path_detector::PathDetector;
use crate::refactor_constraints::generate_constraints;
use crate::semantic_detector::SemanticDetector;
use crate::structural_detector::StructuralDetector;
use crate::types::{AnalysisResult, CallGraphNode};
use petgraph::graph::DiGraph;
use std::collections::HashMap;

/// Main invariant detector that orchestrates all detection strategies
pub struct InvariantDetector<'a> {
    analysis_result: &'a AnalysisResult,
    call_graph: DiGraph<String, ()>,
}

impl<'a> InvariantDetector<'a> {
    /// Create a new invariant detector
    pub fn new(
        analysis_result: &'a AnalysisResult,
        call_graph_nodes: &HashMap<String, CallGraphNode>,
    ) -> Self {
        // Build petgraph from call graph nodes
        let mut graph = DiGraph::new();
        let mut node_indices = HashMap::new();

        // Add all nodes
        for (name, _) in call_graph_nodes {
            let idx = graph.add_node(name.clone());
            node_indices.insert(name.clone(), idx);
        }

        // Add all edges
        for (caller, node) in call_graph_nodes {
            if let Some(&caller_idx) = node_indices.get(caller) {
                for callee in &node.calls {
                    if let Some(&callee_idx) = node_indices.get(callee) {
                        graph.add_edge(caller_idx, callee_idx, ());
                    }
                }
            }
        }

        Self {
            analysis_result,
            call_graph: graph,
        }
    }

    /// Detect all invariants across all detection strategies
    pub fn detect_all(&self) -> InvariantAnalysisResult {
        let mut result = InvariantAnalysisResult::new();

        println!("🔍 Detecting invariants...");

        // 1. Structural invariants (PROVEN)
        println!("  ├─ Detecting structural invariants (graph-based)...");
        let structural_detector = StructuralDetector::new(self.call_graph.clone());
        let structural_invs = structural_detector.detect_all();
        println!("     Found {} structural invariants (from call graph)", structural_invs.len());
        for inv in structural_invs {
            result.add_invariant(inv);
        }

        // 1b. Enhanced structural detection from AnalysisResult
        println!("  ├─ Enhanced structural detection (call graph from analysis)...");
        let enhanced_structural_invs = StructuralDetector::detect_from_analysis(self.analysis_result);
        println!("     Found {} enhanced structural invariants", enhanced_structural_invs.len());
        for inv in enhanced_structural_invs {
            result.add_invariant(inv);
        }

        // 2. Semantic invariants (EMPIRICAL/HEURISTIC)
        println!("  ├─ Detecting semantic invariants...");
        let semantic_detector = SemanticDetector::new(&self.analysis_result.elements);
        let semantic_invs = semantic_detector.detect_all();
        println!("     Found {} semantic invariants", semantic_invs.len());
        for inv in semantic_invs {
            result.add_invariant(inv);
        }

        // 3. Path-intersection invariants (EMPIRICAL)
        println!("  ├─ Detecting path-intersection invariants...");
        let path_detector = PathDetector::new(self.call_graph.clone());
        let path_invs = path_detector.detect_all(10, 100); // max_depth=10, max_paths=100
        println!("     Found {} path-intersection invariants", path_invs.len());
        for inv in path_invs {
            result.add_invariant(inv);
        }

        // 4. Layer assignments
        println!("  ├─ Inferring layers from call graph...");
        let layers = infer_layers(&self.call_graph, 100);
        result.layer_assignments = layers.clone();
        println!("     Inferred {} layer assignments", layers.len());

        // 5. Check for violations
        println!("  └─ Checking for violations...");
        let violations = detect_layer_violations(&layers, &self.call_graph);
        for (caller, callee, caller_layer, callee_layer) in violations {
            let violation = InvariantViolation {
                invariant: Invariant::new(
                    caller.clone(),
                    "".to_string(),
                    InvariantKind::Structural(StructuralInvariant::LayerFixed {
                        layer: caller_layer,
                    }),
                    InvariantStrength::Proven,
                    "Layer violation detected".to_string(),
                ),
                violation_description: format!(
                    "{} (layer {}) calls {} (layer {}) - violation!",
                    caller, caller_layer, callee, callee_layer
                ),
                severity: ViolationSeverity::Critical,
                suggested_fix: Some(format!(
                    "Move {} to layer > {} or refactor call",
                    caller, callee_layer
                )),
            };
            result.add_violation(violation);
        }

        result.stats.update_totals();

        println!();
        println!(
            "✅ Invariants: {} (proven: {}, empirical: {}, heuristic: {})",
            result.stats.total_count,
            result.stats.proven_count,
            result.stats.empirical_count,
            result.stats.heuristic_count
        );
        println!("✅ Violations: {}", result.stats.violation_count);

        result
    }

    /// Check for violations of existing invariants
    #[allow(dead_code)]
    pub fn check_violations(&self, invariants: &[Invariant]) -> Vec<InvariantViolation> {
        let mut violations = Vec::new();

        // Check layer violations
        let layers = infer_layers(&self.call_graph, 100);
        let layer_violations = detect_layer_violations(&layers, &self.call_graph);

        for (caller, callee, caller_layer, callee_layer) in layer_violations {
            // Find the relevant invariant
            if let Some(inv) = invariants.iter().find(|i| i.target == caller) {
                let violation = InvariantViolation {
                    invariant: inv.clone(),
                    violation_description: format!(
                        "{} (layer {}) calls {} (layer {})",
                        caller, caller_layer, callee, callee_layer
                    ),
                    severity: ViolationSeverity::Critical,
                    suggested_fix: Some(format!("Refactor {} to respect layer constraints", caller)),
                };
                violations.push(violation);
            }
        }

        violations
    }

    /// Generate refactoring constraints from detected invariants
    pub fn generate_constraints(&self, result: &InvariantAnalysisResult) -> Vec<crate::refactor_constraints::RefactorConstraint> {
        generate_constraints(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CodeElement, ElementType, Language, Visibility};

    fn make_simple_analysis() -> AnalysisResult {
        let mut result = AnalysisResult::new();

        result.add_element(CodeElement {
            name: "test_fn".to_string(),
            file_path: "test.rs".to_string(),
            line_number: 1,
            element_type: ElementType::Function,
            signature: "fn test_fn() -> i32".to_string(),
            visibility: Visibility::Public,
            generic_params: Vec::new(),
            language: Language::Rust,
            layer: "0".to_string(),
            calls: Vec::new(),
        });

        let mut call_graph = HashMap::new();
        call_graph.insert(
            "test_fn".to_string(),
            CallGraphNode {
                function_name: "test_fn".to_string(),
                file_path: "test.rs".to_string(),
                calls: Vec::new(),
                called_by: Vec::new(),
            },
        );
        result.call_graph = call_graph;

        result
    }

    #[test]
    fn test_invariant_detector_creation() {
        let analysis = make_simple_analysis();
        let detector = InvariantDetector::new(&analysis, &analysis.call_graph);

        assert_eq!(detector.call_graph.node_count(), 1);
    }

    #[test]
    fn test_detect_all() {
        let analysis = make_simple_analysis();
        let detector = InvariantDetector::new(&analysis, &analysis.call_graph);
        let result = detector.detect_all();

        // Should have at least some structural invariants
        assert!(result.stats.structural_count > 0);
    }
}
