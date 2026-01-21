//! Execution Coordinator - bridges executor output to memory input
//!
//! Responsibility:
//! - Receives ExecutionOutcome from executor
//! - Calls propagation engine to normalize deltas
//! - Submits normalized deltas to MemoryEngine for admission
//!
//! Authority: NONE (pure coordination)

use mmsb_proof::{JudgmentProof, AdmissionProof, Hash};
use mmsb_primitives::{PageID, EventId, Timestamp};
use serde::{Deserialize, Serialize};

/// ExecutionCompleted event - emitted after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCompleted {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub judgment_proof: JudgmentProof,
    pub execution_proof_hash: Hash,
    pub proposed_delta_hash: Hash,
    pub page_id: PageID,
}

/// PropagationCompleted event - emitted after propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationCompleted {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub normalized_delta_hash: Hash,
    pub secondary_deltas: Vec<Hash>,
    pub ready_for_admission: bool,
}

/// Coordinator that orchestrates execution → propagation → admission flow
pub struct ExecutionCoordinator;

impl ExecutionCoordinator {
    /// Coordinate the full flow from execution to admission
    /// CRITICAL: This function coordinates but NEVER mutates canonical state
    pub fn coordinate_execution_to_admission(
        judgment: JudgmentProof,
    ) -> CoordinationOutcome {
        // Step 1: Execution happens (produces ExecutionProof + ProposedDelta)
        // Step 2: Propagation normalizes delta (produces NormalizedDelta)
        // Step 3: Memory admits (produces AdmissionProof)
        
        // For now, return stub outcome
        CoordinationOutcome {
            admission_proof: None,
            error: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoordinationOutcome {
    pub admission_proof: Option<AdmissionProof>,
    pub error: Option<String>,
}
