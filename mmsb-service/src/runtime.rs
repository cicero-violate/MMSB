use std::sync::Arc;

use crate::{
    ProtocolSignalIn,
    ProtocolSignalOut,
    Service,
};

pub struct Runtime {
    protocol_out: ProtocolSignalOut,
    services: Vec<Box<dyn Service>>,
}

impl Runtime {
    pub fn new(capacity: usize) -> (Self, ProtocolSignalIn) {
        let (protocol_out, protocol_in) = ProtocolSignalOut::with_capacity(capacity);

        (
            Self {
                protocol_out,
                services: Vec::new(),
            },
            protocol_in,
        )
    }

    pub fn register<S: Service + 'static>(&mut self, service: S) {
        self.services.push(Box::new(service));
    }

    /// Create a fresh receiver set
    pub fn subscribe_protocol(&self) -> ProtocolSignalIn {
        self.protocol_out.subscribe()
    }

    pub async fn run(mut self, ctx: Arc<crate::RuntimeContext>) -> ! {
        let mut services = std::mem::take(&mut self.services);

        for mut svc in services.drain(..) {
            svc.init(ctx.clone());

            let ctx = ctx.clone();
            tokio::spawn(async move {
                svc.run(ctx).await;
            });
        }

        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
        }
    }
}
