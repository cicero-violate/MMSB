//! MMSB Bus Protocol Traits

use mmsb_proof::*;
use crate::AnyEvent;

use crate::events::{IntentCreated, PolicyEvaluated, JudgmentApproved};

/// MMSBSubscription - read-only projection from MMSB
/// Edge-triggered notifications, not callbacks with power
pub trait MMSBSubscription {
    fn subscribe_deltas(&mut self) -> Box<dyn Iterator<Item = Delta>>;
    fn subscribe_events(&mut self) -> Box<dyn Iterator<Item = AnyEvent>>;
    fn project_view(&self, query: StateQuery) -> StateProjection;
}

/// JudgmentProtocol - authority decision chain (A→B→C)
pub trait JudgmentProtocol {
    /// Submit intent, returns IntentCreated event
    fn submit_intent(&mut self, intent: Intent) -> IntentCreated;
    
    /// Evaluate policy, returns PolicyEvaluated event
    fn evaluate_policy(&mut self, event: IntentCreated) -> PolicyEvaluated;
    
    /// Exercise judgment, returns JudgmentApproved if approved
    fn exercise_judgment(&mut self, event: PolicyEvaluated) -> Option<JudgmentApproved>;
    
    /// Write to StateBus (admission request)
    fn request_admission(&mut self, event: JudgmentApproved);
}

/// ExecutionProtocol - mechanical execution of approved actions
pub trait ExecutionProtocol {
    fn execute(&mut self, admission_proof: AdmissionProof) -> ExecutionOutcome;
    fn report_outcome(&mut self, outcome: ExecutionOutcome);
}

/// StateProtocol - SOLE WRITER TO MMSB
pub trait StateProtocol {
    fn admit(&mut self, judgment_proof: JudgmentProof) -> Result<AdmissionProof, AdmissionError>;
    fn commit(&mut self, fact: Fact) -> Result<CommitProof, CommitError>;
    fn broadcast_delta(&self, delta: Delta);
}

/// LearningProtocol - advisory derivation (F→G)
pub trait LearningProtocol {
    fn observe_outcome(&mut self, commit_proof: CommitProof) -> OutcomeProof;
    fn derive_knowledge(&mut self, outcome_proof: OutcomeProof) -> KnowledgeProof;
    fn report_knowledge(&mut self, knowledge_proof: KnowledgeProof);
}

/// ComputeProtocol - GPU/CUDA acceleration
pub trait ComputeProtocol {
    fn compute(&mut self, request: ComputeRequest) -> ComputeResult;
    fn report_result(&mut self, result: ComputeResult);
}

/// ChromiumProtocol - browser automation
pub trait ChromiumProtocol {
    fn execute_command(&mut self, command: BrowserCommand) -> BrowserResult;
    fn report_observation(&mut self, result: BrowserResult);
}

/// ReplayProtocol - historical event stream (read-only, no causality)
pub trait ReplayProtocol: MMSBSubscription {
    fn stream_events(&self, from_epoch: u64, to_epoch: u64) -> EventStream;
    fn replay_to_state(&mut self, target_epoch: u64) -> StateSnapshot;
}

// Placeholder types
pub struct Delta;
pub struct StateQuery;
pub struct StateProjection;
pub struct Intent;
pub struct ExecutionOutcome;
pub struct Fact;
pub struct AdmissionError;
pub struct CommitError;
pub struct ComputeRequest;
pub struct ComputeResult;
pub struct BrowserCommand;
pub struct BrowserResult;
pub struct EventStream;
pub struct StateSnapshot;
