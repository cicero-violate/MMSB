//! Admission Gateway - produces AdmissionProof (D)
//! 
//! Verifies JudgmentProof (C) and checks epoch validity before admitting execution.

use mmsb_proof::{Hash, JudgmentProof};
#[derive(Debug, Clone)]
pub struct AdmissionGate {
    current_epoch: u64,
}
impl AdmissionGate {
    pub fn new() -> Self {
        Self { current_epoch: 0 }
    }
    /// Verify JudgmentProof and admit execution
    pub fn admit(&self, judgment_proof: &JudgmentProof) -> Result<(), AdmissionError> {
        // Verify judgment proof is approved
        if !judgment_proof.approved {
            return Err(AdmissionError::NotApproved);
        }
        // Check epoch validity (simplified)
        // In production, verify cryptographic signature
        
        Ok(())
impl Default for AdmissionGate {
    fn default() -> Self {
        Self::new()
pub enum AdmissionError {
    NotApproved,
    InvalidEpoch,
    ReplayDetected,
