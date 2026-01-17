//! MMSB Proof Authentication
//!
//! This module defines verification of proof correctness and chain integrity.
//! It does NOT make authority decisions - it only validates proof structures.

use mmsb_proof::{Hash, Proof};
use mmsb_proof::{
    AdmissionProof, CommitProof, IntentProof, JudgmentProof, 
    KnowledgeProof, OutcomeProof, PolicyProof
};
use std::fmt;

/// Wrapper enum for any proof type to enable chain verification
#[derive(Debug, Clone)]
pub enum AnyProof {
    Intent(IntentProof),
    Policy(PolicyProof),
    Judgment(JudgmentProof),
    Admission(AdmissionProof),
    Commit(CommitProof),
    Outcome(OutcomeProof),
    Knowledge(KnowledgeProof),
}

impl Proof for AnyProof {
    fn hash(&self) -> Hash {
        match self {
            Self::Intent(p) => p.hash(),
            Self::Policy(p) => p.hash(),
            Self::Judgment(p) => p.hash(),
            Self::Admission(p) => p.hash(),
            Self::Commit(p) => p.hash(),
            Self::Outcome(p) => p.hash(),
            Self::Knowledge(p) => p.hash(),
        }
    }
    
    fn previous(&self) -> Option<Hash> {
        match self {
            Self::Intent(p) => p.previous(),
            Self::Policy(p) => p.previous(),
            Self::Judgment(p) => p.previous(),
            Self::Admission(p) => p.previous(),
            Self::Commit(p) => p.previous(),
            Self::Outcome(p) => p.previous(),
            Self::Knowledge(p) => p.previous(),
        }
    }
}

/// Error types for proof verification
#[derive(Debug, Clone)]
pub enum VerificationError {
    InvalidHash,
    BrokenChain,
    MissingProof,
    InvalidSignature,
    EpochMismatch,
    ReplayDetected,
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHash => write!(f, "Proof hash is invalid"),
            Self::BrokenChain => write!(f, "Proof chain is broken"),
            Self::MissingProof => write!(f, "Required proof is missing"),
            Self::InvalidSignature => write!(f, "Authority signature is invalid"),
            Self::EpochMismatch => write!(f, "Epoch does not match"),
            Self::ReplayDetected => write!(f, "Replay attack detected"),
        }
    }
}

impl std::error::Error for VerificationError {}

/// Generic proof verification trait
pub trait VerifyProof<P: Proof> {
    /// Verify that a proof is structurally valid
    fn verify(proof: &P) -> Result<(), VerificationError>;
}

/// Chain verification trait
pub trait VerifyChain {
    /// Verify that a sequence of proofs forms a valid chain
    /// Each proof must hash-link to the previous one
    fn verify_chain(proofs: &[AnyProof]) -> Result<(), VerificationError>;
}

/// Default chain verifier implementation
pub struct ChainVerifier;

impl VerifyChain for ChainVerifier {
    fn verify_chain(proofs: &[AnyProof]) -> Result<(), VerificationError> {
        if proofs.is_empty() {
            return Ok(());
        }
        
        for i in 1..proofs.len() {
            let prev_hash = proofs[i - 1].hash();
            let current_prev = proofs[i].previous().ok_or(VerificationError::BrokenChain)?;
            
            if prev_hash != current_prev {
                return Err(VerificationError::BrokenChain);
            }
        }
        
        Ok(())
    }
}

/// Verify hash linkage between two proofs
pub fn verify_link<P1: Proof, P2: Proof>(previous: &P1, current: &P2) -> Result<(), VerificationError> {
    let prev_hash = previous.hash();
    let current_prev = current.previous().ok_or(VerificationError::BrokenChain)?;
    
    if prev_hash == current_prev {
        Ok(())
    } else {
        Err(VerificationError::BrokenChain)
    }
}
