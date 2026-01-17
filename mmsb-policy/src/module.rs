//! MMSB Policy Module - implements PolicyStage trait

use mmsb_proof::{
    Hash, IntentCreated, IntentProof, PolicyCategory, PolicyEvaluated, PolicyProof, PolicyStage,
    Proof, ProduceProof, RiskClass,
};
use mmsb_service::{EventBus, EventHandler, Module};

use crate::{IntentClassifier, IntentMetadata};

pub struct PolicyInput {
    pub intent_hash: Hash,
    pub intent_proof: IntentProof,
    pub metadata: IntentMetadata,
}

pub struct PolicyModule {
    event_bus: Option<EventBus>,
    logical_time: u64,
}

impl PolicyModule {
    pub fn new() -> Self {
        Self {
            event_bus: None,
            logical_time: 0,
        }
    }

    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

    fn classify_risk(metadata: &IntentMetadata) -> RiskClass {
        // Simple risk classification based on metadata
        if metadata.files_touched > 10 {
            RiskClass::High
        } else if metadata.diff_lines > 100 {
            RiskClass::Medium
        } else {
            RiskClass::Low
        }
    }

    fn determine_category(metadata: &IntentMetadata) -> PolicyCategory {
        let risk = Self::classify_risk(metadata);
        match risk {
            RiskClass::Low => PolicyCategory::AutoApprove,
            RiskClass::Medium | RiskClass::High | RiskClass::Critical => {
                PolicyCategory::RequiresReview
            }
        }
    }
}

impl Default for PolicyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ProduceProof for PolicyModule {
    type Input = PolicyInput;
    type Proof = PolicyProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let category = Self::determine_category(&input.metadata);
        let risk_class = Self::classify_risk(&input.metadata);
        PolicyProof::new(input.intent_proof.hash(), category, risk_class)
    }
}

impl PolicyStage for PolicyModule {}

impl EventHandler<IntentCreated> for PolicyModule {
    fn on_event(&mut self, event: IntentCreated) {
        let metadata = IntentClassifier::classify("");

        let input = PolicyInput {
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof.clone(),
            metadata,
        };

        let proof = Self::produce_proof(&input);

        let policy_event = PolicyEvaluated {
            event_id: event.intent_hash,
            timestamp: self.next_time(),
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof,
            policy_proof: proof,
        };

        if let Some(bus) = &self.event_bus {
            let _ = bus.emit(mmsb_proof::AnyEvent::PolicyEvaluated(policy_event));
        }
    }
}

impl Module for PolicyModule {
    fn name(&self) -> &str {
        "mmsb-policy"
    }

    fn initialize(&mut self, bus: EventBus) {
        self.event_bus = Some(bus);
    }

    fn shutdown(&mut self) {
        self.event_bus = None;
    }
}
