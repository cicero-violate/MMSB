//! Execution Coordinator - orchestrates execution → propagation → admission
//!
//! Responsibility:
//! - Receives ExecutionOutcome from ExecutionLoop
//! - Coordinates with PropagationEngine to normalize deltas
//! - Submits to MemoryEngine for admission
//!
//! Authority: NONE (pure coordination, never mutates)

use crate::execution_loop::{ExecutionLoop, ExecutionOutcome};
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

/// CoordinationOutcome - result of full execution → propagation → admission
#[derive(Debug, Clone)]
pub struct CoordinationOutcome {
    pub execution_outcome: ExecutionOutcome,
    pub admission_proof: Option<AdmissionProof>,
    pub error: Option<String>,
}

/// ExecutionCoordinator - orchestrates the full pipeline
/// CRITICAL: Coordinates but NEVER mutates canonical state
pub struct ExecutionCoordinator {
    execution_loop: ExecutionLoop,
}

impl ExecutionCoordinator {
    pub fn new() -> Self {
        Self {
            execution_loop: ExecutionLoop::new(),
        }
    }

    /// Coordinate execution of a judgment through the full pipeline
    /// Returns outcome with execution proof and optional admission proof
    /// CRITICAL: This function NEVER mutates canonical state
    pub fn coordinate(
        &mut self,
        judgment: &JudgmentProof,
    ) -> CoordinationOutcome {
        // Phase 1: Execute (produces proof + proposed delta)
        let execution_outcome = self.execution_loop.execute(judgment);

        // Phase 2: Propagation would happen here
        // (normalizes delta, derives secondary deltas)
        // For now, stub

        // Phase 3: Admission would happen here
        // (MemoryEngine verifies and produces AdmissionProof)
        // For now, stub

        CoordinationOutcome {
            execution_outcome,
            admission_proof: None,
            error: None,
        }
    }
}

impl Default for ExecutionCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
