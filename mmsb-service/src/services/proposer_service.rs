// Proposes intents → JudgmentBus

use crate::{RuntimeContext, Service, ServiceContext};
use mmsb_events::JudgmentBus;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct ProposerService {
    name: String,
}

impl ProposerService {
    pub fn new() -> Self {
        Self { name: "proposer".into() }
    }
}

impl Service for ProposerService {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &mut self,
        ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async move {
            ctx.with_judgment_bus(|jb: &mut dyn JudgmentBus| {
                // placeholder intent flow
                // let intent = ...
                // let created = jb.submit_intent(intent);
                // let policy = jb.evaluate_policy(created);
                // let judgment = jb.exercise_judgment(policy);
                // if let Some(j) = judgment {
                //     jb.request_admission(j);
                // }
                let _ = jb;
            });
        })
    }
}
