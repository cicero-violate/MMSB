use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::RuntimeContext;

pub trait Service: Send + Sync {
    fn name(&self) -> &str;

    fn init(&mut self, _ctx: Arc<RuntimeContext>) {}

    fn run(
        &mut self,
        ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>>;
}
