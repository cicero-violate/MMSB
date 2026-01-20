//! MMSB Service Runtime
// This file sends and receives signals, not data and not state

use mmsb_events::{JudgmentApproved, PolicyEvaluated, IntentCreated};
use tokio::sync::broadcast;

#[derive(Clone)]
pub struct EventBus {
    judgment_tx: broadcast::Sender<JudgmentApproved>,
    policy_tx: broadcast::Sender<PolicyEvaluated>,
    intent_tx: broadcast::Sender<IntentCreated>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (judgment_tx, _) = broadcast::channel(capacity);
        let (policy_tx, _) = broadcast::channel(capacity);
        let (intent_tx, _) = broadcast::channel(capacity);
        Self { judgment_tx, policy_tx, intent_tx }
    }
    
    pub fn subscribe_judgment(&self) -> broadcast::Receiver<JudgmentApproved> {
        self.judgment_tx.subscribe()
    }
    
    pub fn subscribe_policy(&self) -> broadcast::Receiver<PolicyEvaluated> {
        self.policy_tx.subscribe()
    }
    
    pub fn subscribe_intent(&self) -> broadcast::Receiver<IntentCreated> {
        self.intent_tx.subscribe()
    }
    
    pub fn emit_judgment(&self, event: JudgmentApproved) {
        let _ = self.judgment_tx.send(event);
    }
    
    pub fn emit_policy(&self, event: PolicyEvaluated) {
        let _ = self.policy_tx.send(event);
    }
    
    pub fn emit_intent(&self, event: IntentCreated) {
        let _ = self.intent_tx.send(event);
    }
}

pub struct Runtime {
    event_bus: EventBus,
}

impl Runtime {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            event_bus: EventBus::new(capacity),
        }
    }
    
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

