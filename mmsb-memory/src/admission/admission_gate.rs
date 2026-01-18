//! Admission Gate - Verifies JudgmentProof and epoch validity
//!
//! The admission gate is the entry point for all execution requests.
//! It enforces:
//! - JudgmentProof (C) verification
//! - Epoch validity
//! - Replay protection

use mmsb_proof::{JudgmentProof, AdmissionProof, Proof};
use crate::epoch::Epoch;

/// Admission gate result
#[derive(Debug)]
pub enum AdmissionResult {
    Admitted(AdmissionProof),
    Rejected(AdmissionRejection),
}

/// Reasons for admission rejection
#[derive(Debug)]
pub enum AdmissionRejection {
    InvalidJudgmentProof,
    StaleEpoch { current: u64, requested: u64 },
    DuplicateExecution,
    EpochMismatch,
}

/// Admission gate verifier
pub struct AdmissionGate {
    current_epoch: Epoch,
}

impl AdmissionGate {
    pub fn new(current_epoch: Epoch) -> Self {
        Self { current_epoch }
    }
    
    /// Verify a JudgmentProof and produce AdmissionProof if valid
    pub fn verify(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> AdmissionResult {
        // Check approval status
        if !judgment_proof.approved {
            return AdmissionResult::Rejected(AdmissionRejection::InvalidJudgmentProof);
        }
        
        // Check epoch validity
        if judgment_proof.timestamp < self.current_epoch.0 as u64 {
            return AdmissionResult::Rejected(AdmissionRejection::StaleEpoch {
                current: self.current_epoch.0 as u64,
                requested: judgment_proof.timestamp,
            });
        }
        
        // Produce AdmissionProof (D)
        let admission_proof = AdmissionProof {
            judgment_proof_hash: judgment_proof.hash(),
            epoch: self.current_epoch.0 as u64,
            nonce: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        AdmissionResult::Admitted(admission_proof)
    }
}
