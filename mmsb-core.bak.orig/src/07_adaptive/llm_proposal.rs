//! LLM-based Structural Proposal Generation
//!
//! Provides the authoritative Phase 7 instruction for LLM agents
//! and enforces Phase 1 judgment criteria.

use crate::dag::DependencyGraph;
use super::propagation_stats::PropagationStats;
use super::types::{StructuralProposal, ProposalCategory};
use super::proposal_validator::{ProposalValidator, ValidationResult};
use serde::{Deserialize, Serialize};

/// LLM Proposal Request
#[derive(Debug, Clone, Serialize)]
pub struct LLMProposalRequest {
    /// Current DAG state (read-only)
    pub dag_summary: DagSummary,
    
    /// Propagation statistics
    pub stats_summary: StatsSummary,
    
    /// System instruction
    pub instruction: String,
}

/// Simplified DAG summary for LLM consumption
#[derive(Debug, Clone, Serialize)]
pub struct DagSummary {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub snapshot_hash: String,
}

/// Simplified stats summary for LLM consumption
#[derive(Debug, Clone, Serialize)]
pub struct StatsSummary {
    pub high_fanout_pages: Vec<(u64, usize)>,
    pub zero_fanout_pages: Vec<u64>,
    pub median_fanout: Option<usize>,
    pub total_propagations: usize,
}

/// LLM Proposal Response
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum LLMProposalResponse {
    Proposal(ProposalJSON),
    NoAction(NoActionResponse),
}

#[derive(Debug, Clone, Deserialize)]
pub struct NoActionResponse {
    pub status: String,
}

/// JSON format for LLM proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalJSON {
    pub proposal_id: String,
    pub ops: Vec<StructuralOpJSON>,
    pub rationale: String,
    pub evidence: EvidenceJSON,
    pub expected_effect: ExpectedEffectJSON,
    pub risk_assessment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StructuralOpJSON {
    AddEdge {
        from: u64,
        to: u64,
        edge_type: String,
    },
    RemoveEdge {
        from: u64,
        to: u64,
    },
    ReplaceEdge {
        old_from: u64,
        old_to: u64,
        new_from: u64,
        new_to: u64,
        new_edge_type: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceJSON {
    pub fanout_before: Option<i64>,
    pub fanout_after_estimate: Option<i64>,
    pub affected_pages: Vec<u64>,
    pub dag_snapshot_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedEffectJSON {
    pub propagation_reduction_pct: Option<i64>,
    pub materialization_change: String,
}

/// Phase 7 LLM instruction generator
pub struct LLMProposalEngine;

impl LLMProposalEngine {
    /// Generate the authoritative Phase 7 instruction
    pub fn generate_instruction() -> String {
        include_str!("phase7_instruction.txt").to_string()
    }
    
    /// Create LLM request from current system state
    pub fn create_request(
        dag: &DependencyGraph,
        stats: &PropagationStats,
    ) -> LLMProposalRequest {
        let dag_summary = DagSummary {
            total_nodes: dag.edges().iter().flat_map(|(f, t, _)| vec![*f, *t]).collect::<std::collections::HashSet<_>>().len(),
            total_edges: dag.edges().len(),
            snapshot_hash: format!("{:x}", dag.version()),
        };
        
        let high_fanout = stats.high_fanout_pages(
            stats.median_fanout().unwrap_or(1) * 2
        );
        
        let stats_summary = StatsSummary {
            high_fanout_pages: high_fanout.iter().map(|(p, c)| (p.0, *c)).collect(),
            zero_fanout_pages: stats.zero_fanout_pages().iter().map(|p| p.0).collect(),
            median_fanout: stats.median_fanout(),
            total_propagations: stats.total_propagations,
        };
        
        LLMProposalRequest {
            dag_summary,
            stats_summary,
            instruction: Self::generate_instruction(),
        }
    }
    
    /// Validate LLM response against Phase 1 criteria
    pub fn validate_response(
        dag: &DependencyGraph,
        response: &ProposalJSON,
    ) -> ValidationResult {
        use crate::dag::{EdgeType, StructuralOp};
        use crate::types::PageID;
        
        // Convert JSON ops to internal format
        let ops: Vec<StructuralOp> = response.ops.iter().filter_map(|op| {
            match op {
                StructuralOpJSON::AddEdge { from, to, edge_type } => {
                    let et = match edge_type.as_str() {
                        "Data" => EdgeType::Data,
                        "Control" => EdgeType::Control,
                        "Gpu" => EdgeType::Gpu,
                        "Compiler" => EdgeType::Compiler,
                        _ => return None,
                    };
                    Some(StructuralOp::AddEdge {
                        from: PageID(*from),
                        to: PageID(*to),
                        edge_type: et,
                    })
                }
                StructuralOpJSON::RemoveEdge { from, to } => {
                    Some(StructuralOp::RemoveEdge {
                        from: PageID(*from),
                        to: PageID(*to),
                    })
                }
                StructuralOpJSON::ReplaceEdge { old_from: _, old_to: _, new_from: _, new_to: _, new_edge_type: _ } => {
                    // Convert to remove + add
                    // For now, skip as it needs multiple ops
                    None
                }
            }
        }).collect();
        
        // Validate structural safety
        let structural = ProposalValidator::validate_structural_safety(dag, &ops);
        if !structural.valid {
            return structural;
        }
        
        // Validate causal justification
        let causal = ProposalValidator::validate_causal_justification(
            response.evidence.fanout_before,
            response.expected_effect.propagation_reduction_pct,
            None,
        );
        if !causal.valid {
            return causal;
        }
        
        // Validate evidence
        let affected: Vec<PageID> = response.evidence.affected_pages.iter().map(|p| PageID(*p)).collect();
        let evidence = ProposalValidator::validate_evidence(
            &affected,
            Some(&response.evidence.dag_snapshot_hash),
        );
        if !evidence.valid {
            return evidence;
        }
        
        ValidationResult::ok()
    }
    
    /// Convert validated JSON proposal to internal format
    pub fn convert_to_proposal(
        response: ProposalJSON,
    ) -> Option<StructuralProposal> {
        use crate::dag::{EdgeType, StructuralOp};
        use crate::types::PageID;
        use super::types::ExpectedEffect;
        
        let ops: Vec<StructuralOp> = response.ops.iter().filter_map(|op| {
            match op {
                StructuralOpJSON::AddEdge { from, to, edge_type } => {
                    let et = match edge_type.as_str() {
                        "Data" => EdgeType::Data,
                        "Control" => EdgeType::Control,
                        "Gpu" => EdgeType::Gpu,
                        "Compiler" => EdgeType::Compiler,
                        _ => return None,
                    };
                    Some(StructuralOp::AddEdge {
                        from: PageID(*from),
                        to: PageID(*to),
                        edge_type: et,
                    })
                }
                StructuralOpJSON::RemoveEdge { from, to } => {
                    Some(StructuralOp::RemoveEdge {
                        from: PageID(*from),
                        to: PageID(*to),
                    })
                }
                StructuralOpJSON::ReplaceEdge { .. } => {
                    // ReplaceEdge requires decomposition into remove + add
                    // Skip for now
                    None
                }
            }
        }).collect();
        
        if ops.is_empty() {
            return None;
        }
        
        let expected_effect = ExpectedEffect {
            propagation_fanout_delta: response.evidence.fanout_after_estimate
                .and_then(|after| response.evidence.fanout_before.map(|before| after - before))
                .unwrap_or(0),
            derived_delta_delta: 0,
            locality_improvement: Some(response.expected_effect.materialization_change.clone()),
        };
        
        let confidence = match response.risk_assessment.to_uppercase().as_str() {
            "LOW" => 0.8,
            "MEDIUM" => 0.5,
            "HIGH" => 0.3,
            _ => 0.5,
        };
        
        Some(StructuralProposal {
            proposal_id: response.proposal_id,
            ops,
            rationale: response.rationale,
            expected_effect,
            confidence,
            category: ProposalCategory::StructuralSimplification,
        })
    }
}
