use std::sync::Arc;
use parking_lot::Mutex;
use std::path::PathBuf;

use mmsb_events::{ExecutionBus, LearningBus, ExecutionOutcome};
use mmsb_proof::{AdmissionProof, CommitProof, OutcomeProof, KnowledgeProof};
use mmsb_memory::{adapter::MemoryAdapter, memory_engine::{MemoryEngine, MemoryEngineConfig}};
use mmsb_memory::page::PageLocation;

use mmsb_service::{
    Runtime,
    RuntimeContext,
    RuntimeScheduler,
    services::ProposerService,
};

// Stub execution and learning buses (to be implemented later)
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
    // Initialize MemoryEngine
    let config = MemoryEngineConfig {
        tlog_path: PathBuf::from("/tmp/mmsb_tlog"),
        default_location: PageLocation::Cpu,
    };
    
    let engine = MemoryEngine::new(config)
        .expect("Failed to initialize MemoryEngine");
    
    // Wrap in Arc<Mutex> for sharing
    let engine = Arc::new(Mutex::new(engine));
    
    // Create adapter that implements both StateBus and MemoryReader
    let adapter = MemoryAdapter::new(engine.clone());
    
    // Split into write and read interfaces
    let state_bus = Arc::new(Mutex::new(adapter));
    // Clone for reader - both point to same adapter/engine
    let memory_reader: Arc<dyn mmsb_events::MemoryReader + Send + Sync> = {
        let adapter = MemoryAdapter::new(engine);
        Arc::new(adapter)
    };
    
    // Initialize runtime
    let (mut runtime, protocol_in) = Runtime::new(1024);

    runtime.register(ProposerService::new());

    let ctx = Arc::new(RuntimeContext::new(
        Arc::new(RuntimeScheduler::default()),
        state_bus,
        memory_reader,
        Arc::new(Mutex::new(StubExecutionBus)),
        Arc::new(Mutex::new(StubLearningBus)),
        protocol_in,
        None,
    ));

    runtime.run(ctx).await;
}
