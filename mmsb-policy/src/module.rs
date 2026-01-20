//! MMSB Policy Module - implements PolicyStage trait

use mmsb_events::{EventSink, IntentCreated, PolicyEvaluated};
use mmsb_proof::{Hash, IntentProof, PolicyCategory, PolicyProof, PolicyStage, Proof, ProduceProof, RiskClass};

pub struct PolicyInput {
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
}

pub struct PolicyModule<S: EventSink> {
    sink: Option<S>,
    logical_time: u64,
}

impl<S: EventSink> PolicyModule<S> {
    pub fn new() -> Self {
        Self {
            sink: None,
            logical_time: 0,
        }
    }

    pub fn with_sink(mut self, sink: S) -> Self {
        self.sink = Some(sink);
        self
    }

    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

   fn classify_risk(_intent_hash: &Hash) -> RiskClass {
       // Simple risk classification - in production, analyze intent content
        // Default: all intents require review unless explicitly in allowed_classes
        RiskClass::Medium
   }

   fn determine_category(risk: RiskClass) -> PolicyCategory {
        match risk {
            RiskClass::Low => PolicyCategory::AutoApprove,
            RiskClass::Medium => PolicyCategory::RequiresReview,
            RiskClass::High
            | RiskClass::Critical => {
                PolicyCategory::RequiresReview
            }
        }
    }
}

impl<S: EventSink> Default for PolicyModule<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: EventSink> ProduceProof for PolicyModule<S> {
    type Input = PolicyInput;
    type Proof = PolicyProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let risk_class = Self::classify_risk(&input.intent_hash);
        let category = Self::determine_category(risk_class.clone());
        PolicyProof::new(input.intent_proof.hash(), category, risk_class)
    }
}

impl<S: EventSink> PolicyStage for PolicyModule<S> {}

impl<S: EventSink> PolicyModule<S> {
    pub fn handle_intent_created(&mut self, event: IntentCreated) {
        let input = PolicyInput {
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof.clone(),
        };

        let proof = Self::produce_proof(&input);

        let policy_event = PolicyEvaluated {
            event_id: event.intent_hash,
            timestamp: self.next_time(),
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof,
            policy_proof: proof,
        };

        if let Some(sink) = &self.sink {
            sink.emit(mmsb_events::AnyEvent::PolicyEvaluated(policy_event));
        }
    }
}
