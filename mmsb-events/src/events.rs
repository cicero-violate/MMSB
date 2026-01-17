//! MMSB Canonical Event Schemas
//!
//! Events are the only mechanism of coordination in MMSB.
//! This module defines all canonical event types per the specification.

use serde::{Deserialize, Serialize};
use std::fmt;

use mmsb_proof::{
    AdmissionProof, CommitProof, Hash, IntentProof, JudgmentProof,
    KnowledgeProof, OutcomeProof, PolicyProof,
};

/// Event ID - content-addressed unique identifier
pub type EventId = Hash;

/// Logical timestamp (not wall-clock authority)
pub type Timestamp = u64;

/// Marker trait for all events in MMSB
pub trait Event: fmt::Debug + Clone + Send + Sync {
    fn event_type(&self) -> EventType;
    fn event_id(&self) -> EventId;
    fn timestamp(&self) -> Timestamp;
}

/// Canonical event types in MMSB
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    IntentCreated,
    PolicyEvaluated,
    JudgmentApproved,
    ExecutionRequested,
    MemoryCommitted,
    KnowledgeDerived,
}

// ============================================================================
// Event 1: IntentCreated
// ============================================================================

/// IntentCreated - emitted by mmsb-intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCreated {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
}

impl Event for IntentCreated {
    fn event_type(&self) -> EventType {
        EventType::IntentCreated
    }
    
    fn event_id(&self) -> EventId {
        self.event_id
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

// ============================================================================
// Event 2: PolicyEvaluated
// ============================================================================

/// PolicyEvaluated - emitted by mmsb-policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluated {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
    pub policy_proof: PolicyProof,
}

impl Event for PolicyEvaluated {
    fn event_type(&self) -> EventType {
        EventType::PolicyEvaluated
    }
    
    fn event_id(&self) -> EventId {
        self.event_id
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

// ============================================================================
// Event 3: JudgmentApproved
// ============================================================================

/// JudgmentApproved - emitted by mmsb-judgment (AUTHORITY WITNESS)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentApproved {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub policy_proof: PolicyProof,
    pub judgment_proof: JudgmentProof,
}

impl Event for JudgmentApproved {
    fn event_type(&self) -> EventType {
        EventType::JudgmentApproved
    }
    
    fn event_id(&self) -> EventId {
        self.event_id
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

// ============================================================================
// Event 4: ExecutionRequested (updated)
// ============================================================================

/// ExecutionRequested - emitted by mmsb-executor when ready to apply an approved plan
// In mmsb-events/src/events.rs

// ExecutionRequested — what executor sends to trigger commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequested {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    
    pub judgment_proof: JudgmentProof,          // from mmsb-proof (already allowed)
    
    // Instead of full Delta:
    pub delta_hash: Hash,                       // hash of the delta to apply
    pub delta_size_hint: Option<u64>,           // optional, helps memory estimate resources
    
    // Minimal info for propagation (small, serializable)
    pub affected_page_ids: Vec<PageID>,         // ← still needed, but PageID must be defined in mmsb-proof or a shared primitive crate
}



impl Event for ExecutionRequested {
    fn event_type(&self) -> EventType { EventType::ExecutionRequested }
    fn event_id(&self) -> EventId { self.event_id }
    fn timestamp(&self) -> Timestamp { self.timestamp }
}

// ============================================================================
// Event 5: MemoryCommitted (updated)
// ============================================================================
// MemoryCommitted — what memory emits after successful commit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCommitted {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    
    pub delta_hash: Hash,
    pub epoch: u64,
    pub snapshot_ref: Option<Hash>,
    
    pub admission_proof: AdmissionProof,
    pub commit_proof: CommitProof,
    pub outcome_proof: OutcomeProof,
    
    // What executor needs for physical propagation
    pub affected_page_ids: Vec<PageID>,
}


impl Event for MemoryCommitted {
    fn event_type(&self) -> EventType { EventType::MemoryCommitted }
    fn event_id(&self) -> EventId { self.event_id }
    fn timestamp(&self) -> Timestamp { self.timestamp }
}

// ============================================================================
// Event 6: KnowledgeDerived (PERSISTED)
// ============================================================================

/// KnowledgeDerived - emitted by mmsb-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeDerived {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_ref: Hash,
    pub execution_plan_ref: Hash,
    pub outcome_summary: Hash,
    pub outcome_proof: OutcomeProof,
    pub knowledge_proof: KnowledgeProof,
}

impl Event for KnowledgeDerived {
    fn event_type(&self) -> EventType {
        EventType::KnowledgeDerived
    }
    
    fn event_id(&self) -> EventId {
        self.event_id
    }
    
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

// ============================================================================
// Event Envelope Wrapper
// ============================================================================

/// Generic event envelope for any event type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyEvent {
    IntentCreated(IntentCreated),
    PolicyEvaluated(PolicyEvaluated),
    JudgmentApproved(JudgmentApproved),
    ExecutionRequested(ExecutionRequested),
    MemoryCommitted(MemoryCommitted),
    KnowledgeDerived(KnowledgeDerived),
}

impl Event for AnyEvent {
    fn event_type(&self) -> EventType {
        match self {
            Self::IntentCreated(e) => e.event_type(),
            Self::PolicyEvaluated(e) => e.event_type(),
            Self::JudgmentApproved(e) => e.event_type(),
            Self::ExecutionRequested(e) => e.event_type(),
            Self::MemoryCommitted(e) => e.event_type(),
            Self::KnowledgeDerived(e) => e.event_type(),
        }
    }
    
    fn event_id(&self) -> EventId {
        match self {
            Self::IntentCreated(e) => e.event_id(),
            Self::PolicyEvaluated(e) => e.event_id(),
            Self::JudgmentApproved(e) => e.event_id(),
            Self::ExecutionRequested(e) => e.event_id(),
            Self::MemoryCommitted(e) => e.event_id(),
            Self::KnowledgeDerived(e) => e.event_id(),
        }
    }
    
    fn timestamp(&self) -> Timestamp {
        match self {
            Self::IntentCreated(e) => e.timestamp(),
            Self::PolicyEvaluated(e) => e.timestamp(),
            Self::JudgmentApproved(e) => e.timestamp(),
            Self::ExecutionRequested(e) => e.timestamp(),
            Self::MemoryCommitted(e) => e.timestamp(),
            Self::KnowledgeDerived(e) => e.timestamp(),
        }
    }
}
