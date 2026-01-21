// Executes approved work → ExecutionBus

use crate::{RuntimeContext, Service, ServiceContext};
use mmsb_events::ExecutionBus;
use mmsb_proof::AdmissionProof;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct ExecutorService {
    name: String,
}

impl ExecutorService {
    pub fn new() -> Self {
        Self { name: "executor".into() }
    }

    fn execute_one(&self, ctx: &RuntimeContext, admission: AdmissionProof) {
        ctx.with_execution_bus(|eb: &mut dyn ExecutionBus| {
            let outcome = eb.execute(admission);
            eb.report_outcome(outcome);
        });
    }
}

impl Service for ExecutorService {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &mut self,
        _ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async {})
    }
}
