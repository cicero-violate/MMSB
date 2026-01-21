// Commits truth → StateBus (sole writer)

use crate::{RuntimeContext, Service, ServiceContext};
use mmsb_events::StateBus;
use mmsb_events::Fact;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct CommitService {
    name: String,
}

impl CommitService {
    pub fn new() -> Self {
        Self { name: "commit".into() }
    }

    pub fn commit(&self, ctx: &RuntimeContext, fact: Fact) {
        ctx.with_state_bus(|sb: &mut dyn StateBus| {
            let _commit_proof = sb.commit(fact).expect("commit failed");
        });
        // CommitPulse fires inside mmsb-memory
    }
}

impl Service for CommitService {
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
