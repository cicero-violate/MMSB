#![allow(unused_imports)]
//! Phase 7: Adaptive Structural Proposals
//!
//! **PHASE LAW COMPLIANCE:**
//! - This phase is READ-ONLY
//! - NO mutations of DAG or state
//! - NO propagation triggers
//! - NO automatic application
//! - All outputs are discardable proposals
//!
//! **Purpose:**
//! Generate structural improvement proposals based on observed system behavior.
//! All proposals must route through Phase 1 (structural judgment) for approval.

pub mod types;
pub mod propagation_stats;
pub mod proposal_engine;
pub mod proposal_validator;
pub mod llm_proposal;

// Phase 7 public API
pub use types::{
    ExpectedEffect, ProposalCategory, ProposalError, ProposalID, StructuralProposal,
};
pub use propagation_stats::PropagationStats;
pub use proposal_engine::{ProposalConfig, ProposalEngine};

// Legacy exports (deprecated)
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub use crate::optimization::{AccessPattern, MemoryLayout, PageId, PhysAddr};
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub use crate::optimization::{PageCluster, PageClusterer};
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub use crate::optimization::LocalityOptimizer;
pub use proposal_validator::{ProposalValidator, ValidationResult, ValidationViolation};
pub use llm_proposal::{
    LLMProposalEngine, LLMProposalRequest, LLMProposalResponse,
    ProposalJSON, StructuralOpJSON, EvidenceJSON, ExpectedEffectJSON,
};
