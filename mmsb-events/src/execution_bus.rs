//! ExecutionBus - Mechanical execution flow

use mmsb_proof::*;
use mmsb_primitives::{Hash, Timestamp, EventId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub success: bool,
    pub result_hash: Hash,
}

pub trait ExecutionBus {
    fn execute(&mut self, admission_proof: AdmissionProof) -> ExecutionOutcome;
    fn report_outcome(&mut self, outcome: ExecutionOutcome);
}
