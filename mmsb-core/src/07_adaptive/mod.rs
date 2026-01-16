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

// Legacy modules (to be deprecated)
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub mod memory_layout;
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub mod page_clustering;
#[deprecated(note = "Legacy code, use proposal_engine instead")]
pub mod locality_optimizer;

// Phase 7 public API
pub use types::{
    ExpectedEffect, ProposalCategory, ProposalError, ProposalID, StructuralProposal,
};
pub use propagation_stats::PropagationStats;
pub use proposal_engine::{ProposalConfig, ProposalEngine};

// Legacy exports (deprecated)
#[allow(deprecated)]
pub use memory_layout::{MemoryLayout, AccessPattern, PageId, PhysAddr};
#[allow(deprecated)]
pub use page_clustering::{PageCluster, PageClusterer};
#[allow(deprecated)]
pub use locality_optimizer::LocalityOptimizer;
