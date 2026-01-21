//! Event Listener Service - Example of consuming commit events
//!
//! This demonstrates the ZERO-POLLING push model:
//! - Service awaits events from memory
//! - Gets notified instantly when commits occur
//! - No wasted CPU cycles

use crate::{RuntimeContext, Service};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub struct EventListenerService {
    name: String,
}

impl EventListenerService {
    pub fn new() -> Self {
        Self { name: "event_listener".into() }
    }
}

impl Service for EventListenerService {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(
        &mut self,
        ctx: Arc<RuntimeContext>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        Box::pin(async move {
            // Subscribe to commit events (zero polling!)
            let mut events = ctx.memory_reader().subscribe_commits();
            
            println!("📡 EventListenerService: Subscribed to commit events");
            
            // Event loop - waits for events (no CPU waste!)
            let mut count = 0;
            while let Ok(event) = events.recv().await {
                count += 1;
                println!("🎉 EventListenerService: Received commit #{}", count);
                println!("   - Epoch: {}", event.epoch);
                println!("   - Event ID: {:?}", event.event_id);
                println!("   - Pages affected: {}", event.affected_page_ids.len());
                println!("   - Commit successful: {}", event.outcome_proof.success);
                
                // Service could do work here:
                // - Execute admitted work
                // - Learn from outcomes
                // - Update caches
                // - Notify external systems
            }
            
            println!("📡 EventListenerService: Event stream ended");
        })
    }
}
