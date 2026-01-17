//! Phase 1 Judgment Criteria Validator
//!
//! Enforces structural safety and causal justification rules before proposals
//! are submitted to Phase 1 judgment.

use crate::dag::{DependencyGraph, EdgeType, StructuralOp};
use crate::types::PageID;
use std::collections::{HashMap, HashSet};

/// Validation result for a structural proposal
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<ValidationViolation>,
}

/// Types of validation violations
#[derive(Debug, Clone)]
pub enum ValidationViolation {
    /// Proposal would create a cycle
    WouldCreateCycle,
    
    /// Proposal would create orphaned pages
    WouldCreateOrphans(Vec<PageID>),
    
    /// Proposal would create self-dependency
    SelfDependency(PageID),
    
    /// Too many operations (>5)
    TooManyOperations(usize),
    
    /// No causal justification provided
    NoCausalJustification,
    
    /// Missing required evidence
    MissingEvidence(String),
    
    /// Speculative edge without justification
    SpeculativeEdge(PageID, PageID),
}

impl ValidationResult {
    pub fn ok() -> Self {
        Self {
            valid: true,
            violations: Vec::new(),
        }
    }
    
    pub fn fail(violation: ValidationViolation) -> Self {
        Self {
            valid: false,
            violations: vec![violation],
        }
    }
    
    pub fn with_violations(violations: Vec<ValidationViolation>) -> Self {
        Self {
            valid: violations.is_empty(),
            violations,
        }
    }
}

/// Phase 1 judgment criteria validator
pub struct ProposalValidator;

impl ProposalValidator {
    /// Validate structural safety of operations
    pub fn validate_structural_safety(
        dag: &DependencyGraph,
        ops: &[StructuralOp],
    ) -> ValidationResult {
        let mut violations = Vec::new();
        
        // Rule: Max 5 operations
        if ops.len() > 5 {
            violations.push(ValidationViolation::TooManyOperations(ops.len()));
        }
        
        // Simulate applying operations
        let mut simulated_edges = dag.edges();
        
        for op in ops {
            match op {
                StructuralOp::AddEdge { from, to, edge_type } => {
                    // Check for self-dependency
                    if from == to {
                        violations.push(ValidationViolation::SelfDependency(*from));
                    }
                    
                    simulated_edges.push((*from, *to, *edge_type));
                }
                StructuralOp::RemoveEdge { from, to } => {
                    simulated_edges.retain(|(f, t, _)| !(f == from && t == to));
                }
            }
        }
        
        // Check for cycles in simulated graph
        if Self::has_cycle(&simulated_edges) {
            violations.push(ValidationViolation::WouldCreateCycle);
        }
        
        // Check for orphans
        let orphans = Self::find_orphans(&simulated_edges);
        if !orphans.is_empty() {
            violations.push(ValidationViolation::WouldCreateOrphans(orphans));
        }
        
        ValidationResult::with_violations(violations)
    }
    
    /// Check if edge set contains cycles
    fn has_cycle(edges: &[(PageID, PageID, EdgeType)]) -> bool {
        let mut adjacency: HashMap<PageID, Vec<PageID>> = HashMap::new();
        
        for (from, to, _) in edges {
            adjacency.entry(*from).or_default().push(*to);
        }
        
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for node in adjacency.keys() {
            if Self::has_cycle_dfs(*node, &adjacency, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        
        false
    }
    
    fn has_cycle_dfs(
        node: PageID,
        adjacency: &HashMap<PageID, Vec<PageID>>,
        visited: &mut HashSet<PageID>,
        rec_stack: &mut HashSet<PageID>,
    ) -> bool {
        if rec_stack.contains(&node) {
            return true;
        }
        
        if visited.contains(&node) {
            return false;
        }
        
        visited.insert(node);
        rec_stack.insert(node);
        
        if let Some(neighbors) = adjacency.get(&node) {
            for neighbor in neighbors {
                if Self::has_cycle_dfs(*neighbor, adjacency, visited, rec_stack) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(&node);
        false
    }
    
    /// Find pages that would become orphaned (no incoming or outgoing edges)
    fn find_orphans(edges: &[(PageID, PageID, EdgeType)]) -> Vec<PageID> {
        let mut connected: HashSet<PageID> = HashSet::new();
        
        for (from, to, _) in edges {
            connected.insert(*from);
            connected.insert(*to);
        }
        
        // Note: In real usage, we'd need the full node set to detect true orphans
        // This is a placeholder - actual implementation would compare against DAG nodes
        Vec::new()
    }
    
    /// Validate causal justification exists
    pub fn validate_causal_justification(
        expected_fanout_reduction: Option<i64>,
        expected_delta_reduction: Option<i64>,
        locality_improvement: Option<&str>,
    ) -> ValidationResult {
        let has_justification = expected_fanout_reduction.is_some()
            || expected_delta_reduction.is_some()
            || locality_improvement.is_some();
        
        if !has_justification {
            return ValidationResult::fail(ValidationViolation::NoCausalJustification);
        }
        
        ValidationResult::ok()
    }
    
    /// Validate evidence is provided
    pub fn validate_evidence(
        affected_pages: &[PageID],
        dag_snapshot_hash: Option<&str>,
    ) -> ValidationResult {
        let mut violations = Vec::new();
        
        if affected_pages.is_empty() {
            violations.push(ValidationViolation::MissingEvidence(
                "No affected pages specified".to_string(),
            ));
        }
        
        if dag_snapshot_hash.is_none() {
            violations.push(ValidationViolation::MissingEvidence(
                "No DAG snapshot hash specified".to_string(),
            ));
        }
        
        ValidationResult::with_violations(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detects_self_dependency() {
        let dag = DependencyGraph::new();
        let ops = vec![StructuralOp::AddEdge {
            from: PageID(1),
            to: PageID(1),
            edge_type: EdgeType::Data,
        }];
        
        let result = ProposalValidator::validate_structural_safety(&dag, &ops);
        assert!(!result.valid);
        assert!(matches!(
            result.violations[0],
            ValidationViolation::SelfDependency(_)
        ));
    }
    
    #[test]
    fn test_rejects_too_many_ops() {
        let dag = DependencyGraph::new();
        let ops: Vec<StructuralOp> = (0..6)
            .map(|i| StructuralOp::RemoveEdge {
                from: PageID(i),
                to: PageID(i + 1),
            })
            .collect();
        
        let result = ProposalValidator::validate_structural_safety(&dag, &ops);
        assert!(!result.valid);
        assert!(matches!(
            result.violations[0],
            ValidationViolation::TooManyOperations(6)
        ));
    }
    
    #[test]
    fn test_requires_causal_justification() {
        let result = ProposalValidator::validate_causal_justification(None, None, None);
        assert!(!result.valid);
        assert!(matches!(
            result.violations[0],
            ValidationViolation::NoCausalJustification
        ));
    }
    
    #[test]
    fn test_requires_evidence() {
        let result = ProposalValidator::validate_evidence(&[], None);
        assert!(!result.valid);
        assert_eq!(result.violations.len(), 2);
    }
}
