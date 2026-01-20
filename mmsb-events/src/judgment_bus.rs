//! JudgmentBus - Authority decision chain
//! Flow: Intent (A) → Policy (B) → Judgment (C) → StateBus

use mmsb_proof::*;
use mmsb_primitives::{Hash, Timestamp, EventId};
use serde::{Deserialize, Serialize};

// ============================================================================
// JudgmentBus Structs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub description: String,
    pub intent_class: String,
    pub target_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub files_touched: usize,
    pub diff_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCreated {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
    pub intent_class: String,
    pub target_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub files_touched: usize,
    pub diff_lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluated {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
    pub policy_proof: PolicyProof,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentApproved {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub intent_hash: Hash,
    pub policy_proof: PolicyProof,
    pub judgment_proof: JudgmentProof,
}

// ============================================================================
// JudgmentBus Protocol Trait
// ============================================================================

pub trait JudgmentBus {
    /// Submit intent, returns IntentCreated
    fn submit_intent(&mut self, intent: Intent) -> IntentCreated;
    
    /// Evaluate policy, returns PolicyEvaluated
    fn evaluate_policy(&mut self, event: IntentCreated) -> PolicyEvaluated;
    
    /// Exercise judgment authority, returns JudgmentApproved if approved
    fn exercise_judgment(&mut self, event: PolicyEvaluated) -> Option<JudgmentApproved>;
    
    /// Write to StateBus (admission request)
    fn request_admission(&mut self, event: JudgmentApproved);
}
