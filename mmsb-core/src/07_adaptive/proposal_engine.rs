//! Phase 7 Proposal Engine (Read-Only, Advisory)
//!
//! Generates structural improvement proposals based on observed system behavior.
//!
//! **CRITICAL INVARIANTS:**
//! - This module is READ-ONLY
//! - NO DAG mutations
//! - NO state mutations
//! - NO propagation triggers
//! - NO automatic application
//! - All outputs are discardable proposals

use crate::dag::{DependencyGraph, EdgeType, StructuralOp};
use crate::types::PageID;
use super::propagation_stats::PropagationStats;
use super::types::{
    ExpectedEffect, ProposalCategory, ProposalID, StructuralProposal,
};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Phase 7 Proposal Engine
///
/// Analyzes DAG structure and propagation behavior to propose improvements.
/// **Does not mutate anything.**
pub struct ProposalEngine {
    /// Configuration thresholds
    config: ProposalConfig,
}

/// Configuration for proposal generation
#[derive(Debug, Clone)]
pub struct ProposalConfig {
    /// Fanout threshold for "high fanout" detection
    pub high_fanout_threshold_multiplier: f32,
    
    /// Minimum confidence to emit a proposal
    pub min_confidence: f32,
    
    /// Maximum proposals per category
    pub max_proposals_per_category: usize,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            high_fanout_threshold_multiplier: 2.0,
            min_confidence: 0.3,
            max_proposals_per_category: 10,
        }
    }
}

impl ProposalEngine {
    /// Create new proposal engine
    pub fn new(config: ProposalConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ProposalConfig::default())
    }
    
    /// Generate structural proposals from DAG and stats
    ///
    /// **READ-ONLY:** Does not mutate DAG or state.
    pub fn generate_proposals(
        &self,
        dag: &DependencyGraph,
        stats: &PropagationStats,
    ) -> Vec<StructuralProposal> {
        let mut proposals = Vec::new();
        
        // Category 1: High-fanout reduction
        proposals.extend(self.propose_high_fanout_reduction(dag, stats));
        
        // Category 2: Dead dependency elimination
        proposals.extend(self.propose_dead_dependency_elimination(dag, stats));
        
        // Category 3: Structural simplification
        proposals.extend(self.propose_structural_simplification(dag, stats));
        
        // Category 4: Locality optimization
        proposals.extend(self.propose_locality_optimization(dag, stats));
        
        // Filter by confidence
        proposals.retain(|p| p.confidence >= self.config.min_confidence);
        
        proposals
    }
    
    /// Category 1: Propose high-fanout reduction
    fn propose_high_fanout_reduction(
        &self,
        dag: &DependencyGraph,
        stats: &PropagationStats,
    ) -> Vec<StructuralProposal> {
        let mut proposals = Vec::new();
        
        let median = stats.median_fanout().unwrap_or(1);
        let threshold = (median as f32 * self.config.high_fanout_threshold_multiplier) as usize;
        
        let high_fanout = stats.high_fanout_pages(threshold);
        
        for (page, fanout) in high_fanout.iter().take(self.config.max_proposals_per_category) {
            let descendants = dag.descendants(*page);
            
            if descendants.len() > 3 {
                // Propose: introduce intermediate node to split fanout
                let proposal_id = Uuid::new_v4().to_string();
                
                // Create a hypothetical intermediate node
                // (Phase 1 would actually allocate this if approved)
                // Use a placeholder ID (Phase 1 will assign real ID if approved)
                let intermediate_id = PageID(u64::MAX - page.0);
                
                let mut ops = Vec::new();
                
                // Add edge: root → intermediate
                ops.push(StructuralOp::AddEdge {
                    from: *page,
                    to: intermediate_id,
                    edge_type: EdgeType::Data,
                });
                
                // Split descendants: half through intermediate
                let split_point = descendants.len() / 2;
                for (i, descendant) in descendants.iter().enumerate() {
                    if i < split_point {
                        // Move edge through intermediate
                        ops.push(StructuralOp::RemoveEdge {
                            from: *page,
                            to: *descendant,
                        });
                        ops.push(StructuralOp::AddEdge {
                            from: intermediate_id,
                            to: *descendant,
                            edge_type: EdgeType::Data,
                        });
                    }
                }
                
                let confidence = if *fanout > threshold * 2 { 0.8 } else { 0.5 };
                
                proposals.push(StructuralProposal {
                    proposal_id,
                    ops,
                    rationale: format!(
                        "Page {} has high fanout ({} > threshold {}). Introduce intermediate node to reduce propagation breadth.",
                        page, fanout, threshold
                    ),
                    expected_effect: ExpectedEffect {
                        propagation_fanout_delta: -(split_point as i64),
                        derived_delta_delta: 0,
                        locality_improvement: Some("Split high-fanout node".to_string()),
                    },
                    confidence,
                    category: ProposalCategory::HighFanoutReduction,
                });
            }
        }
        
        proposals
    }
    
    /// Category 2: Propose dead dependency elimination
    fn propose_dead_dependency_elimination(
        &self,
        dag: &DependencyGraph,
        stats: &PropagationStats,
    ) -> Vec<StructuralProposal> {
        let mut proposals = Vec::new();
        
        let zero_fanout = stats.zero_fanout_pages();
        
        for page in zero_fanout.iter().take(self.config.max_proposals_per_category) {
            let edges = dag.edges();
            let outgoing: Vec<_> = edges
                .iter()
                .filter(|(from, _, _)| from == page)
                .collect();
            
            if !outgoing.is_empty() {
                let proposal_id = Uuid::new_v4().to_string();
                
                let ops: Vec<StructuralOp> = outgoing
                    .iter()
                    .map(|(from, to, _)| StructuralOp::RemoveEdge {
                        from: *from,
                        to: *to,
                    })
                    .collect();
                
                proposals.push(StructuralProposal {
                    proposal_id,
                    ops,
                    rationale: format!(
                        "Page {} has never triggered downstream propagation. Consider removing unused dependencies.",
                        page
                    ),
                    expected_effect: ExpectedEffect {
                        propagation_fanout_delta: 0,
                        derived_delta_delta: -(outgoing.len() as i64),
                        locality_improvement: None,
                    },
                    confidence: 0.6,
                    category: ProposalCategory::DeadDependencyElimination,
                });
            }
        }
        
        proposals
    }
    
    /// Category 3: Propose structural simplification
    fn propose_structural_simplification(
        &self,
        dag: &DependencyGraph,
        _stats: &PropagationStats,
    ) -> Vec<StructuralProposal> {
        let mut proposals = Vec::new();
        
        // Detect diamond patterns: A → B → D, A → C → D
        let edges = dag.edges();
        let mut adjacency: HashMap<PageID, Vec<(PageID, EdgeType)>> = HashMap::new();
        
        for (from, to, edge_type) in edges {
            adjacency.entry(from).or_default().push((to, edge_type));
        }
        
        let mut diamonds_found = 0;
        
        for (node_a, children) in &adjacency {
            if children.len() >= 2 {
                // Check if children converge to common descendant
                let child_descendants: Vec<HashSet<PageID>> = children
                    .iter()
                    .map(|(child, _)| dag.descendants(*child))
                    .collect();
                
                if child_descendants.len() >= 2 {
                    let common: HashSet<PageID> = child_descendants[0]
                        .intersection(&child_descendants[1])
                        .copied()
                        .collect();
                    
                    if !common.is_empty() && diamonds_found < self.config.max_proposals_per_category {
                        // Found diamond pattern
                        let proposal_id = Uuid::new_v4().to_string();
                        
                        // Propose: remove one redundant path
                        let (child_b, _) = children[1];
                        
                        let ops = vec![StructuralOp::RemoveEdge {
                            from: *node_a,
                            to: child_b,
                        }];
                        
                        proposals.push(StructuralProposal {
                            proposal_id,
                            ops,
                            rationale: format!(
                                "Detected diamond pattern at {}. Redundant path creates duplicate propagation.",
                                node_a
                            ),
                            expected_effect: ExpectedEffect {
                                propagation_fanout_delta: -(common.len() as i64),
                                derived_delta_delta: -(common.len() as i64),
                                locality_improvement: Some("Eliminate redundant path".to_string()),
                            },
                            confidence: 0.7,
                            category: ProposalCategory::StructuralSimplification,
                        });
                        
                        diamonds_found += 1;
                    }
                }
            }
        }
        
        proposals
    }
    
    /// Category 4: Propose locality optimization
    fn propose_locality_optimization(
        &self,
        _dag: &DependencyGraph,
        _stats: &PropagationStats,
    ) -> Vec<StructuralProposal> {
        // Placeholder: Locality optimization requires additional metadata
        // (e.g., physical memory addresses, cache miss rates)
        // This would be integrated with Phase 6 utility telemetry.
        
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DependencyGraph;
    
    #[test]
    fn test_proposal_engine_does_not_mutate_dag() {
        let mut dag = DependencyGraph::new();
        let page_a = PageID::new();
        let page_b = PageID::new();
        
        // Apply an operation directly to create test structure
        dag.apply_structural_op(&StructuralOp::AddEdge {
            from: page_a,
            to: page_b,
            edge_type: EdgeType::Data,
        }).ok();
        
        let version_before = dag.version();
        let edges_before = dag.edges().len();
        
        let engine = ProposalEngine::default();
        let stats = PropagationStats::new();
        
        let _proposals = engine.generate_proposals(&dag, &stats);
        
        // Verify DAG unchanged
        assert_eq!(dag.version(), version_before);
        assert_eq!(dag.edges().len(), edges_before);
    }
}
