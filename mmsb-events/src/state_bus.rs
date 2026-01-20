//! StateBus - SOLE WRITER TO MMSB

use mmsb_proof::*;
use mmsb_primitives::{Hash, Timestamp, EventId, PageID};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub fact_hash: Hash,
    pub timestamp: Timestamp,
    pub content: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AdmissionError(pub String);

#[derive(Debug, Clone)]
pub struct CommitError(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub delta_id: u64,
    pub changes: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCommitted {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub commit_proof: CommitProof,
    pub delta_hash: Hash,
    pub epoch: u64,
    pub snapshot_ref: Hash,
    pub admission_proof: AdmissionProof,
    pub outcome_proof: OutcomeProof,
    pub affected_page_ids: Vec<PageID>,
}

pub trait StateBus {
    fn admit(&mut self, judgment_proof: JudgmentProof) -> Result<AdmissionProof, AdmissionError>;
    fn commit(&mut self, fact: Fact) -> Result<CommitProof, CommitError>;
    fn broadcast_delta(&self, delta: Delta);
}
