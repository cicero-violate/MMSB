//! MMSB Judgment Module - SOLE AUTHORITY

use mmsb_events::{JudgmentApproved, PolicyEvaluated, JudgmentBus, IntentCreated, Intent};
use mmsb_proof::{Hash, JudgmentProof, JudgmentStage, PolicyProof, Proof, ProduceProof};

pub struct JudgmentInput {
    pub intent_hash: Hash,
    pub policy_proof: PolicyProof,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub intent_hash: Hash,
    pub approved_operations: Vec<String>,
    pub constraints: Vec<String>,
}

pub struct JudgmentModule {
    logical_time: u64,
}

impl JudgmentModule {
    pub fn new() -> Self {
        Self { logical_time: 0 }
    }

    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

    fn make_judgment(policy_proof: &PolicyProof) -> bool {
        match &policy_proof.category {
            mmsb_proof::PolicyCategory::AutoApprove => true,
            mmsb_proof::PolicyCategory::RequiresReview => {
                matches!(policy_proof.risk_class, mmsb_proof::RiskClass::Low | mmsb_proof::RiskClass::Medium)
            }
            mmsb_proof::PolicyCategory::Denied => false,
        }
    }

    fn create_execution_plan(intent_hash: Hash) -> ExecutionPlan {
        ExecutionPlan {
            intent_hash,
            approved_operations: vec!["execute_intent".to_string()],
            constraints: vec!["respect_bounds".to_string()],
        }
    }

    pub fn handle_policy_evaluated(&mut self, event: PolicyEvaluated) -> Option<JudgmentApproved> {
        let input = JudgmentInput {
            intent_hash: event.intent_hash,
            policy_proof: event.policy_proof.clone(),
        };

        let judgment_proof = Self::produce_proof(&input);
        
        if judgment_proof.approved {
            Some(JudgmentApproved {
                event_id: event.intent_hash,
                timestamp: self.next_time(),
                intent_hash: event.intent_hash,
                policy_proof: event.policy_proof,
                judgment_proof,
            })
        } else {
            None
        }
    }
}

impl Default for JudgmentModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ProduceProof for JudgmentModule {
    type Input = JudgmentInput;
    type Proof = JudgmentProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let approved = Self::make_judgment(&input.policy_proof);
        let mut authority_signature = [0u8; 64];
        authority_signature[0] = if approved { 1 } else { 0 };
        
        JudgmentProof::new(
            input.policy_proof.hash(),
            approved,
            authority_signature,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
    }
}

impl JudgmentStage for JudgmentModule {}

impl JudgmentBus for JudgmentModule {
    fn submit_intent(&mut self, _intent: Intent) -> IntentCreated {
        unimplemented!("Intent submission in mmsb-intent")
    }
    
    fn evaluate_policy(&mut self, _event: IntentCreated) -> PolicyEvaluated {
        unimplemented!("Policy evaluation in mmsb-policy")
    }
    
    fn exercise_judgment(&mut self, event: PolicyEvaluated) -> Option<JudgmentApproved> {
        self.handle_policy_evaluated(event)
    }
    
    fn request_admission(&mut self, _event: JudgmentApproved) {
        // Write to StateBus (implemented in runtime)
    }
}
