//! MMSB Service Runtime
//!
//! Provides event bus, module loader, and replay mechanism.
//! This runtime has ZERO authority - it only routes events.

use mmsb_proof::{AnyEvent, Event};
use tokio::sync::broadcast;

/// Event handler trait - modules implement this to react to events
pub trait EventHandler<E: Event>: Send + Sync {
    fn on_event(&mut self, event: E);
}

/// Event emitter trait - modules use this to emit events
pub trait EmitEvent<E: Event> {
    fn emit(&self, event: E);
}

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

/// Module loader trait - modules implement this for initialization
pub trait Module: Send + Sync {
    /// Module name for identification
    fn name(&self) -> &str;
    
    /// Initialize the module with the event bus
    fn initialize(&mut self, bus: EventBus);
    
    /// Shutdown the module gracefully
    fn shutdown(&mut self);
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
        module.initialize(self.event_bus.clone());
        self.modules.push(module);
    }
    
    /// Get a reference to the event bus
    pub fn event_bus(&self) -> &EventBus {
        &self.event_bus
    }
    
    /// Shutdown all modules
    pub fn shutdown(&mut self) {
        for module in &mut self.modules {
            module.shutdown();
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay engine for reconstructing state from events
pub struct ReplayEngine {
    events: Vec<AnyEvent>,
}

impl ReplayEngine {
    /// Create a new replay engine
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }
    
    /// Add an event to the replay log
    pub fn record(&mut self, event: AnyEvent) {
        self.events.push(event);
    }
    
    /// Replay all events through the provided event bus
    pub fn replay(&self, bus: &EventBus) -> Result<usize, broadcast::error::SendError<AnyEvent>> {
        let mut count = 0;
        for event in &self.events {
            bus.emit(event.clone())?;
            count += 1;
        }
        Ok(count)
    }
    
    /// Get the number of recorded events
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
    
    /// Clear all recorded events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for ReplayEngine {
    fn default() -> Self {
        Self::new()
    }
}
