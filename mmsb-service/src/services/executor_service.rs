// Executes approved work → ExecutionBus
// executor_service
// Consumes JudgmentProof
// Cannot re-evaluate policy
// Cannot invent intent


use crate::{RuntimeContext, Service, ServiceContext};
use mmsb_events::{ExecutionBus, ExecutionOutcome as ExecutionBusOutcome};
use mmsb_proof::{AdmissionProof, JudgmentProof};
use mmsb_executor::ExecutionLoop;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use parking_lot::Mutex;

pub struct ExecutorService {
    name: String,
    execution_loop: Arc<Mutex<ExecutionLoop>>,
}

impl ExecutorService {
    pub fn new() -> Self {
        Self {
            name: "executor".into(),
            execution_loop: Arc::new(Mutex::new(ExecutionLoop::new())),
        }
    }

    /// Execute approved judgment
    /// CRITICAL: This never mutates canonical state - only produces ExecutionProof
    fn execute_judgment(&self, ctx: &RuntimeContext, judgment: &JudgmentProof) {
        // Perform mechanical execution (produces proof + proposed delta)
        let outcome = {
            let mut exec_loop = self.execution_loop.lock();
            exec_loop.execute(judgment)
        };

        // Report outcome to execution bus (for propagation + admission)
        ctx.with_execution_bus(|eb: &mut dyn ExecutionBus| {
            let bus_outcome = ExecutionBusOutcome {
                event_id: outcome.proof.execution_id,
                timestamp: outcome.proof.timestamp,
                success: outcome.proof.success,
                result_hash: outcome.proof.result_hash,
            };
            eb.report_outcome(bus_outcome);
        });
    }
}

impl Service for ExecutorService {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &mut self,
        ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        let execution_loop = self.execution_loop.clone();
        
        Box::pin(async move {
            // Subscribe to MemoryCommitted events from memory engine
            let mut events = ctx.memory_reader().subscribe_commits();
            
            loop {
                match events.recv().await {
                    Ok(commit_event) => {
                        // Extract JudgmentProof from admission proof chain
                        // In production: traverse proof chain to get original judgment
                        
                        // For now, we need to reconstruct or lookup the judgment
                        // This is where we'd integrate with the judgment lookup system
                        
                        // Execute the work
                        // self.execute_judgment(&ctx, &judgment);
                    }
                    Err(_) => {
                        // Channel closed, exit loop
                        break;
                    }
                }
            }
        })
    }
}
