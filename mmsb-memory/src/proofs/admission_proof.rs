//! AdmissionProof (D) - produced by mmsb-memory

use mmsb_proof::{AdmissionProof, Hash};
pub struct AdmissionProofBuilder;
impl AdmissionProofBuilder {
    pub fn new(judgment_proof_hash: Hash, epoch: u64) -> AdmissionProof {
        AdmissionProof::new(judgment_proof_hash, epoch, true)
    }
}
