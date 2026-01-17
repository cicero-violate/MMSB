//! MMSB Service Runtime
//!
//! Provides event bus and module wiring.
//! This runtime has ZERO authority - it only routes events.

use mmsb_events::{AnyEvent, EventSink, Module};
use tokio::sync::broadcast;

/// Event bus for routing events between modules
#[derive(Clone)]
pub struct EventBus {
    sender: broadcast::Sender<AnyEvent>,
}

impl EventBus {
    /// Create a new event bus with specified capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }
    
    /// Subscribe to all events on this bus
    pub fn subscribe(&self) -> broadcast::Receiver<AnyEvent> {
        self.sender.subscribe()
    }
    
    /// Emit an event to all subscribers
    pub fn emit(&self, event: AnyEvent) -> Result<usize, broadcast::error::SendError<AnyEvent>> {
        self.sender.send(event)
    }
    
    /// Get the number of active receivers
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl EventSink for EventBus {
    fn emit(&self, event: AnyEvent) {
        let _ = self.sender.send(event);
    }
}

/// MMSB Service Runtime
pub struct Runtime {
    event_bus: EventBus,
    modules: Vec<Box<dyn Module>>,
}

impl Runtime {
    /// Create a new runtime with default event bus capacity
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }
    
    /// Create a new runtime with specified event bus capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            event_bus: EventBus::new(capacity),
            modules: Vec::new(),
        }
    }
    
    /// Register a module with the runtime
    pub fn register_module(&mut self, mut module: Box<dyn Module>) {
        module.attach(Box::new(self.event_bus.clone()));
        self.modules.push(module);
    }
    
    /// Get a reference to the event bus
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }
    
    /// Shutdown all modules
    pub fn shutdown(&mut self) {
        for module in &mut self.modules {
            module.detach();
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
