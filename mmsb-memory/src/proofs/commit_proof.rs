//! CommitProof (E) - produced by mmsb-memory

use mmsb_proof::{CommitProof, Hash};

pub struct CommitProofBuilder;

impl CommitProofBuilder {
    pub fn new(admission_proof_hash: Hash, delta_hash: Hash) -> CommitProof {
        CommitProof::new(admission_proof_hash, delta_hash, true)
    }
}
