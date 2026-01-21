use std::sync::Arc;
use parking_lot::Mutex;
use std::path::PathBuf;

use mmsb_events::{ExecutionBus, LearningBus, ExecutionOutcome};
use mmsb_proof::{AdmissionProof, CommitProof, OutcomeProof, KnowledgeProof};
use mmsb_memory::{
    adapter::MemoryAdapter,
    memory_engine::{MemoryEngine, MemoryEngineConfig},
    notifier::CommitNotifier,
    page::PageLocation,
};

use mmsb_service::{
    Runtime,
    RuntimeContext,
    RuntimeScheduler,
    services::{ProposerService, EventListenerService},
};

// TODO: Replace with real implementation
struct PlaceholderExecutionBus;
impl ExecutionBus for PlaceholderExecutionBus {
    fn execute(&mut self, _admission: AdmissionProof) -> ExecutionOutcome {
        ExecutionOutcome {
            event_id: [0u8; 32],
            timestamp: 0,
            success: true,
            result_hash: [0u8; 32],
        }
    }
    fn report_outcome(&mut self, _outcome: ExecutionOutcome) {
        // Placeholder - will be replaced with real implementation
    }
}

struct PlaceholderLearningBus;
impl LearningBus for PlaceholderLearningBus {
    fn observe_outcome(&mut self, _: CommitProof) -> OutcomeProof {
        OutcomeProof {
            commit_proof_hash: [0u8; 32],
            success: true,
            error_class: None,
            rollback_hash: None,
        }
    }
    fn derive_knowledge(&mut self, _: OutcomeProof) -> KnowledgeProof {
        KnowledgeProof {
            outcome_proof_hash: [0u8; 32],
            knowledge_hash: [0u8; 32],
            confidence: 1.0,
        }
    }
    fn report_knowledge(&mut self, _: KnowledgeProof) {
        // Placeholder
    }
}

#[tokio::main]
async fn main() {
    println!("🚀 Starting MMSB Service Runtime...");
    
    // Step 1: Create CommitNotifier (event infrastructure)
    let (notifier, _initial_rx) = CommitNotifier::new(1024);
    let notifier = Arc::new(notifier);
    println!("✅ Created CommitNotifier (capacity: 1024)");
    
    // Step 2: Initialize MemoryEngine with notifier
    let config = MemoryEngineConfig {
        tlog_path: PathBuf::from("/tmp/mmsb_tlog"),
        default_location: PageLocation::Cpu,
    };
    
    let engine = MemoryEngine::new(config, notifier.clone())
        .expect("Failed to initialize MemoryEngine");
    
    let engine = Arc::new(Mutex::new(engine));
    println!("✅ Initialized MemoryEngine");
    
    // Step 3: Create adapters (both use same notifier!)
    let write_adapter = MemoryAdapter::new(engine.clone(), notifier.clone());
    let read_adapter = MemoryAdapter::new(engine, notifier);
    
    // Step 4: Split into interfaces
    let state_bus = Arc::new(Mutex::new(write_adapter));
    let memory_reader: Arc<dyn mmsb_events::MemoryReader + Send + Sync> = Arc::new(read_adapter);
    println!("✅ Created MemoryAdapter (StateBus + MemoryReader)");
    
    // Step 5: Initialize runtime
    let (mut runtime, protocol_in) = Runtime::new(1024);

    runtime.register(ProposerService::new());
    runtime.register(EventListenerService::new());
    println!("✅ Registered services");

    let ctx = Arc::new(RuntimeContext::new(
        Arc::new(RuntimeScheduler::default()),
        state_bus,
        memory_reader,
        Arc::new(Mutex::new(PlaceholderExecutionBus)),
        Arc::new(Mutex::new(PlaceholderLearningBus)),
        protocol_in,
        None,
    ));
    
    println!("✅ RuntimeContext created");
    println!("🎯 Services can now:");
    println!("   - Subscribe to commits: ctx.memory_reader().subscribe_commits()");
    println!("   - Query state: ctx.memory_reader().current_epoch()");
    println!("   - Write via: ctx.with_state_bus(...)");
    println!("\n🔥 Running services (Ctrl+C to stop)...\n");

    runtime.run(ctx).await;
}
