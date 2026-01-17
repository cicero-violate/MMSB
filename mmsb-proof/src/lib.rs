//! MMSB Proof System
//!
//! This module defines all canonical proof structures (A → G) used in MMSB.
//! Proofs are immutable artifacts that witness checks and decisions.
//! They do NOT carry authority.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::fmt;

/// Type alias for proof hashes
pub type Hash = [u8; 32];

/// Marker trait for all proofs in the MMSB system
pub trait Proof: fmt::Debug + Clone + Send + Sync {
    /// Compute stable hash of this proof
    fn hash(&self) -> Hash;
    
    /// Reference to previous proof in chain (if any)
    fn previous(&self) -> Option<Hash>;
}

/// Generic proof production role
pub trait ProduceProof {
    type Input;
    type Proof: Proof;

    fn produce_proof(input: &Self::Input) -> Self::Proof;
}

// ============================================================================
// Proof Stage Marker Traits (A → G)
// ============================================================================

/// A — Intent Stage: produces IntentProof
pub trait IntentStage: ProduceProof<Proof = IntentProof> {}

/// B — Policy Stage: produces PolicyProof
pub trait PolicyStage: ProduceProof<Proof = PolicyProof> {}

/// C — Judgment Stage: produces JudgmentProof (AUTHORITY WITNESS)
pub trait JudgmentStage: ProduceProof<Proof = JudgmentProof> {}

/// D — Admission Stage: produces AdmissionProof
pub trait AdmissionStage: ProduceProof<Proof = AdmissionProof> {}

/// E — Commit Stage: produces CommitProof
pub trait CommitStage: ProduceProof<Proof = CommitProof> {}

/// F — Outcome Stage: produces OutcomeProof
pub trait OutcomeStage: ProduceProof<Proof = OutcomeProof> {}

/// G — Knowledge Stage: produces KnowledgeProof
pub trait KnowledgeStage: ProduceProof<Proof = KnowledgeProof> {}

// ============================================================================
// A — IntentProof
// ============================================================================

/// IntentProof (A): "This intent is well-formed, canonical, and bounded."
///
/// Produced by: mmsb-intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentProof {
    /// Canonical hash of the intent
    pub intent_hash: Hash,
    
    /// Intent schema version
    pub schema_version: u32,
    
    /// Declared resource bounds
    pub bounds: IntentBounds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentBounds {
    pub max_duration_ms: u64,
    pub max_memory_bytes: u64,
}

impl IntentProof {
    pub fn new(intent_hash: Hash, schema_version: u32, bounds: IntentBounds) -> Self {
        Self {
            intent_hash,
            schema_version,
            bounds,
        }
    }
}

impl Proof for IntentProof {
    fn hash(&self) -> Hash {
        // Simple hash implementation - in production use proper crypto hash
        self.intent_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        None // First in chain
    }
}

// ============================================================================
// B — PolicyProof
// ============================================================================

/// PolicyProof (B): "This intent was classified under policy rules."
///
/// Produced by: mmsb-policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyProof {
    /// Hash of IntentProof (A)
    pub intent_proof_hash: Hash,
    
    /// Policy classification
    pub category: PolicyCategory,
    
    /// Risk assessment
    pub risk_class: RiskClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyCategory {
    AutoApprove,
    RequiresReview,
    Denied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskClass {
    Low,
    Medium,
    High,
    Critical,
}

impl PolicyProof {
    pub fn new(intent_proof_hash: Hash, category: PolicyCategory, risk_class: RiskClass) -> Self {
        Self {
            intent_proof_hash,
            category,
            risk_class,
        }
    }
}

impl Proof for PolicyProof {
    fn hash(&self) -> Hash {
        self.intent_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.intent_proof_hash)
    }
}

// ============================================================================
// C — JudgmentProof (AUTHORITY WITNESS)
// ============================================================================

/// JudgmentProof (C): "This intent and policy are approved for execution."
///
/// Produced by: mmsb-judgment (SOLE AUTHORITY)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentProof {
    /// Hash of PolicyProof (B)
    pub policy_proof_hash: Hash,
    
    /// Explicit approval decision
    pub approved: bool,
    
    /// Authority signature
    #[serde(with = "BigArray")]
    pub authority_signature: [u8; 64],
    
    /// Timestamp of judgment
    pub timestamp: u64,
}

impl JudgmentProof {
    pub fn new(
        policy_proof_hash: Hash,
        approved: bool,
        authority_signature: [u8; 64],
        timestamp: u64,
    ) -> Self {
        Self {
            policy_proof_hash,
            approved,
            authority_signature,
            timestamp,
        }
    }
}

impl Proof for JudgmentProof {
    fn hash(&self) -> Hash {
        self.policy_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.policy_proof_hash)
    }
}

// ============================================================================
// D — AdmissionProof
// ============================================================================

/// AdmissionProof (D): "This execution was admitted under valid judgment."
///
/// Produced by: mmsb-memory (pre-commit gate)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionProof {
    /// Hash of JudgmentProof (C)
    pub judgment_proof_hash: Hash,
    
    /// Epoch in which admission occurred
    pub epoch: u64,
    
    /// Replay protection nonce
    pub nonce: u64,
}

impl AdmissionProof {
    pub fn new(judgment_proof_hash: Hash, epoch: u64, nonce: u64) -> Self {
        Self {
            judgment_proof_hash,
            epoch,
            nonce,
        }
    }
}

impl Proof for AdmissionProof {
    fn hash(&self) -> Hash {
        self.judgment_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.judgment_proof_hash)
    }
}

// ============================================================================
// E — CommitProof
// ============================================================================

/// CommitProof (E): "The mutation occurred and all invariants held."
///
/// Produced by: mmsb-memory (commit)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitProof {
    /// Hash of AdmissionProof (D)
    pub admission_proof_hash: Hash,
    
    /// Hash of applied delta
    pub delta_hash: Hash,
    
    /// State hash after commit
    pub state_hash: Hash,
    
    /// Invariants verified
    pub invariants_held: bool,
}

impl CommitProof {
    pub fn new(
        admission_proof_hash: Hash,
        delta_hash: Hash,
        state_hash: Hash,
        invariants_held: bool,
    ) -> Self {
        Self {
            admission_proof_hash,
            delta_hash,
            state_hash,
            invariants_held,
        }
    }
}

impl Proof for CommitProof {
    fn hash(&self) -> Hash {
        self.admission_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.admission_proof_hash)
    }
}

// ============================================================================
// F — OutcomeProof
// ============================================================================

/// OutcomeProof (F): "This was the observable result of the commit."
///
/// Produced by: mmsb-memory (post-commit)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeProof {
    /// Hash of CommitProof (E)
    pub commit_proof_hash: Hash,
    
    /// Success or failure
    pub success: bool,
    
    /// Error classification (if failed)
    pub error_class: Option<ErrorClass>,
    
    /// Rollback record hash (if any)
    pub rollback_hash: Option<Hash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorClass {
    InvariantViolation,
    ResourceExhaustion,
    InvalidState,
    Unknown,
}

impl OutcomeProof {
    pub fn new(
        commit_proof_hash: Hash,
        success: bool,
        error_class: Option<ErrorClass>,
        rollback_hash: Option<Hash>,
    ) -> Self {
        Self {
            commit_proof_hash,
            success,
            error_class,
            rollback_hash,
        }
    }
}

impl Proof for OutcomeProof {
    fn hash(&self) -> Hash {
        self.commit_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.commit_proof_hash)
    }
}

// ============================================================================
// G — KnowledgeProof
// ============================================================================

/// KnowledgeProof (G): "This outcome was learned and recorded."
///
/// Produced by: mmsb-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProof {
    /// Hash of OutcomeProof (F)
    pub outcome_proof_hash: Hash,
    
    /// Derived pattern or correlation
    pub pattern_hash: Hash,
    
    /// Risk signal (non-authoritative)
    pub risk_signal: f64,
}

impl KnowledgeProof {
    pub fn new(outcome_proof_hash: Hash, pattern_hash: Hash, risk_signal: f64) -> Self {
        Self {
            outcome_proof_hash,
            pattern_hash,
            risk_signal,
        }
    }
}

impl Proof for KnowledgeProof {
    fn hash(&self) -> Hash {
        self.outcome_proof_hash
    }
    
    fn previous(&self) -> Option<Hash> {
        Some(self.outcome_proof_hash)
    }
}

// ============================================================================
// Event System
// ============================================================================
