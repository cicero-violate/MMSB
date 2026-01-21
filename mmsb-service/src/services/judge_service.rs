// Policy / judgment authority (pure JudgmentBus choreography)
// judge_service
// Runs the full pipeline:
// mmsb-intent (normalize / validate intent)
// mmsb-policy (evaluate constraints)
// mmsb-judgment (issue JudgmentProof)
// Is the only service allowed to produce judgments


use crate::{RuntimeContext, Service};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct JudgeService {
    name: String,
}

impl JudgeService {
    pub fn new() -> Self {
        Self { name: "judge".into() }
    }
}

impl Service for JudgeService {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &mut self,
        _ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
            }
        })
    }
}
