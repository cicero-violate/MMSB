//! JudgmentBus Events - Intent → Policy → Judgment → Memory

use serde::{Deserialize, Serialize};
use crate::*;
use mmsb_primitives::{Hash, EventId, Timestamp};
use mmsb_proof::{IntentProof, PolicyProof, JudgmentProof};

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

impl Event for IntentCreated {
    fn event_type(&self) -> EventType { EventType::IntentCreated }
    fn event_id(&self) -> EventId { self.event_id }
    fn timestamp(&self) -> Timestamp { self.timestamp }
}

impl Event for PolicyEvaluated {
    fn event_type(&self) -> EventType { EventType::PolicyEvaluated }
    fn event_id(&self) -> EventId { self.event_id }
    fn timestamp(&self) -> Timestamp { self.timestamp }
}

impl Event for JudgmentApproved {
    fn event_type(&self) -> EventType { EventType::JudgmentApproved }
    fn event_id(&self) -> EventId { self.event_id }
    fn timestamp(&self) -> Timestamp { self.timestamp }
}
