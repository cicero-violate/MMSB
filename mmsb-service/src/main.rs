use std::sync::{Arc, Mutex};

use mmsb_events::{StateBus, ExecutionBus, LearningBus, Fact, ExecutionOutcome};
use mmsb_proof::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof, KnowledgeProof};

use mmsb_service::{
    Runtime,
    RuntimeContext,
    RuntimeScheduler,
    services::ProposerService,
};

// Stub implementations for compilation
struct StubStateBus;
impl StateBus for StubStateBus {
    fn admit(&mut self, _: JudgmentProof) -> Result<AdmissionProof, mmsb_events::state_bus::AdmissionError> {
        unimplemented!("StubStateBus::admit")
    }
    fn commit(&mut self, _: Fact) -> Result<CommitProof, mmsb_events::state_bus::CommitError> {
        unimplemented!("StubStateBus::commit")
    }
    fn broadcast_delta(&self, _: mmsb_events::state_bus::Delta) {
        unimplemented!("StubStateBus::broadcast_delta")
    }
}

struct StubExecutionBus;
impl ExecutionBus for StubExecutionBus {
    fn execute(&mut self, _: AdmissionProof) -> ExecutionOutcome {
        unimplemented!("StubExecutionBus::execute")
    }
    fn report_outcome(&mut self, _: ExecutionOutcome) {
        unimplemented!("StubExecutionBus::report_outcome")
    }
}

struct StubLearningBus;
impl LearningBus for StubLearningBus {
    fn observe_outcome(&mut self, _: CommitProof) -> OutcomeProof {
        unimplemented!("StubLearningBus::observe_outcome")
    }
    fn derive_knowledge(&mut self, _: OutcomeProof) -> KnowledgeProof {
        unimplemented!("StubLearningBus::derive_knowledge")
    }
    fn report_knowledge(&mut self, _: KnowledgeProof) {
        unimplemented!("StubLearningBus::report_knowledge")
    }
}

#[tokio::main]
async fn main() {
    let (mut runtime, protocol_in) = Runtime::new(1024);

    runtime.register(ProposerService::new());

    let ctx = Arc::new(RuntimeContext::new(
        Arc::new(RuntimeScheduler::default()),
        Arc::new(Mutex::new(StubStateBus)),
        Arc::new(Mutex::new(StubExecutionBus)),
        Arc::new(Mutex::new(StubLearningBus)),
        protocol_in,
        None,
    ));

    runtime.run(ctx).await;
}
