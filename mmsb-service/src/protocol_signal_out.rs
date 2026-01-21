use tokio::sync::broadcast;
use mmsb_events::{JudgmentApproved, PolicyEvaluated, IntentCreated};

pub struct ProtocolSignalOut {
    judgment_tx: broadcast::Sender<JudgmentApproved>,
    policy_tx: broadcast::Sender<PolicyEvaluated>,
    intent_tx: broadcast::Sender<IntentCreated>,
}

impl ProtocolSignalOut {
    pub fn with_capacity(capacity: usize) -> (Self, crate::ProtocolSignalIn) {
        let (judgment_tx, _) = broadcast::channel(capacity);
        let (policy_tx, _) = broadcast::channel(capacity);
        let (intent_tx, _) = broadcast::channel(capacity);

        let out = Self {
            judgment_tx,
            policy_tx,
            intent_tx,
        };

        let input = out.subscribe();

        (out, input)
    }

    pub fn subscribe(&self) -> crate::ProtocolSignalIn {
        crate::ProtocolSignalIn {
            judgment_rx: self.judgment_tx.subscribe(),
            policy_rx: self.policy_tx.subscribe(),
            intent_rx: self.intent_tx.subscribe(),
        }
    }

    // emitters (examples)
    pub fn emit_judgment(&self, j: JudgmentApproved) {
        let _ = self.judgment_tx.send(j);
    }

    pub fn emit_policy(&self, p: PolicyEvaluated) {
        let _ = self.policy_tx.send(p);
    }

    pub fn emit_intent(&self, i: IntentCreated) {
        let _ = self.intent_tx.send(i);
    }
}
