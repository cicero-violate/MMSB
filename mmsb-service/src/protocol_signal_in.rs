use tokio::sync::broadcast;
use mmsb_events::{JudgmentApproved, PolicyEvaluated, IntentCreated};

pub struct ProtocolSignalIn {
    pub(crate) judgment_rx: broadcast::Receiver<JudgmentApproved>,
    pub(crate) policy_rx: broadcast::Receiver<PolicyEvaluated>,
    pub(crate) intent_rx: broadcast::Receiver<IntentCreated>,
}

impl ProtocolSignalIn {
    pub fn try_judgment(&mut self) -> Option<JudgmentApproved> {
        self.judgment_rx.try_recv().ok()
    }

    pub fn try_policy(&mut self) -> Option<PolicyEvaluated> {
        self.policy_rx.try_recv().ok()
    }

    pub fn try_intent(&mut self) -> Option<IntentCreated> {
        self.intent_rx.try_recv().ok()
    }
}
