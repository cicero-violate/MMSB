// Observes outcomes → LearningBus (advisory only)

use crate::{RuntimeContext, Service, ServiceContext};
use mmsb_events::LearningBus;
use mmsb_proof::CommitProof;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct LearningService {
    name: String,
}

impl LearningService {
    pub fn new() -> Self {
        Self { name: "learning".into() }
    }

    pub fn learn(&self, ctx: &RuntimeContext, commit: CommitProof) {
        ctx.with_learning_bus(|lb: &mut dyn LearningBus| {
            let outcome = lb.observe_outcome(commit);
            let knowledge = lb.derive_knowledge(outcome);
            lb.report_knowledge(knowledge);
        });
    }
}

impl Service for LearningService {
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
