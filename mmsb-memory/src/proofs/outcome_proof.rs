//! OutcomeProof (F) - produced by mmsb-memory

use mmsb_proof::{OutcomeProof, Hash};

pub struct OutcomeProofBuilder;

impl OutcomeProofBuilder {
    pub fn new(commit_proof_hash: Hash, success: bool) -> OutcomeProof {
        OutcomeProof::new(commit_proof_hash, success, String::new())
    }
    
    pub fn with_error(commit_proof_hash: Hash, error: String) -> OutcomeProof {
        OutcomeProof::new(commit_proof_hash, false, error)
    }
}
