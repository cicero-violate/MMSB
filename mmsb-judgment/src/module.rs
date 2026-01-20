//! MMSB Judgment Module - implements JudgmentStage trait
//! 
//! This module holds SOLE AUTHORITY in the MMSB system.
//! It exercises explicit approval authority over execution.

use mmsb_events::{EventSink, JudgmentApproved, PolicyEvaluated};
use mmsb_proof::{Hash, JudgmentProof, JudgmentStage, PolicyProof, Proof, ProduceProof};

/// Input required to produce JudgmentProof
pub struct JudgmentInput {
    pub intent_hash: Hash,
    pub policy_proof: PolicyProof,
}

/// Execution plan approved by judgment
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub intent_hash: Hash,
    pub approved_operations: Vec<String>,
    pub constraints: Vec<String>,
}

/// JudgmentModule - SOLE AUTHORITY
/// 
/// This module is the only component that can approve execution.
/// It produces JudgmentProof (C) witnessing explicit approval.
pub struct JudgmentModule<S: EventSink> {
    sink: Option<S>,
    logical_time: u64,
}

impl<S: EventSink> JudgmentModule<S> {
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

    /// Exercise judgment authority - decide whether to approve
    fn make_judgment(policy_proof: &PolicyProof) -> bool {
        // Authority decision: approve based on policy classification
        match &policy_proof.category {
            mmsb_proof::PolicyCategory::AutoApprove => true,
            mmsb_proof::PolicyCategory::RequiresReview => {
                // In this simple implementation, approve medium risk
                // In production, this would involve actual review
                matches!(policy_proof.risk_class, mmsb_proof::RiskClass::Low | mmsb_proof::RiskClass::Medium)
            }
            mmsb_proof::PolicyCategory::Denied => false,
        }
    }

    /// Create execution plan for approved judgments
    fn create_execution_plan(intent_hash: Hash) -> ExecutionPlan {
        ExecutionPlan {
            intent_hash,
            approved_operations: vec!["execute_intent".to_string()],
            constraints: vec!["respect_bounds".to_string()],
        }
    }
}

impl<S: EventSink> Default for JudgmentModule<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: EventSink> ProduceProof for JudgmentModule<S> {
    type Input = JudgmentInput;
    type Proof = JudgmentProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let approved = Self::make_judgment(&input.policy_proof);
        
        // Authority signature (simplified - in production use real crypto)
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

impl<S: EventSink> JudgmentStage for JudgmentModule<S> {}

impl<S: EventSink> JudgmentModule<S> {
    pub fn handle_policy_evaluated(&mut self, event: PolicyEvaluated) {
        let input = JudgmentInput {
            intent_hash: event.intent_hash,
            policy_proof: event.policy_proof.clone(),
        };

        let judgment_proof = Self::produce_proof(&input);
        
        // Only emit JudgmentApproved if actually approved
        if judgment_proof.approved {
            let judgment_event = JudgmentApproved {
                event_id: event.intent_hash,
                timestamp: self.next_time(),
                intent_hash: event.intent_hash,
                policy_proof: event.policy_proof,
                judgment_proof,
            };

            if let Some(sink) = &self.sink {
                sink.emit(mmsb_events::AnyEvent::JudgmentApproved(judgment_event));
            }
        }
    }
}

// Implement JudgmentProtocol
use mmsb_events::{JudgmentProtocol, IntentCreated, Intent};

impl<S: EventSink> JudgmentProtocol for JudgmentModule<S> {
    fn submit_intent(&mut self, _intent: Intent) -> IntentCreated {
        unimplemented!("Intent submission happens in mmsb-intent")
    }
    
    fn evaluate_policy(&mut self, _event: IntentCreated) -> PolicyEvaluated {
        unimplemented!("Policy evaluation happens in mmsb-policy")
    }
    
    fn exercise_judgment(&mut self, event: PolicyEvaluated) -> Option<JudgmentApproved> {
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
    
    fn request_admission(&mut self, event: JudgmentApproved) {
        if let Some(sink) = &self.sink {
            sink.emit(event);
        }
    }
}
