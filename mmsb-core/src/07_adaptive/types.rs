//! Phase 7 Types — Structural Proposals
//!
//! All types in this module are **read-only observations** and **discardable proposals**.
//! Nothing here mutates system state.

use crate::dag::{EdgeType, StructuralOp};
use crate::types::PageID;
use serde::{Deserialize, Serialize};

/// Unique identifier for a proposal
pub type ProposalID = String;

/// A structural improvement proposal (Phase 7 output)
///
/// **CRITICAL:** This is a proposal only. It does NOT mutate the DAG.
/// All proposals must route through Phase 1 (structural judgment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralProposal {
    /// Unique identifier for this proposal
    pub proposal_id: ProposalID,
    
    /// Structural operations to apply (if approved)
    pub ops: Vec<StructuralOp>,
    
    /// Human-readable rationale
    pub rationale: String,
    
    /// Expected system effect
    pub expected_effect: ExpectedEffect,
    
    /// Confidence score [0.0, 1.0]
    pub confidence: f32,
    
    /// Proposal category
    pub category: ProposalCategory,
}

/// Expected impact of applying a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedEffect {
    /// Change in propagation fanout (negative = improvement)
    pub propagation_fanout_delta: i64,
    
    /// Change in derived delta count (negative = improvement)
    pub derived_delta_delta: i64,
    
    /// Optional locality improvement description
    pub locality_improvement: Option<String>,
}

/// Allowed proposal categories (exhaustive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalCategory {
    /// Reduce high-fanout propagation
    HighFanoutReduction,
    
    /// Remove unused dependencies
    DeadDependencyElimination,
    
    /// Simplify DAG structure
    StructuralSimplification,
    
    /// Improve memory locality
    LocalityOptimization,
}

impl StructuralProposal {
    /// Validate proposal invariants
    pub fn validate(&self) -> Result<(), ProposalError> {
        if self.ops.is_empty() {
            return Err(ProposalError::EmptyOps);
        }
        
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(ProposalError::InvalidConfidence(self.confidence));
        }
        
        Ok(())
    }
}

/// Proposal validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ProposalError {
    #[error("proposal contains no operations")]
    EmptyOps,
    
    #[error("confidence score {0} out of range [0.0, 1.0]")]
    InvalidConfidence(f32),
}
