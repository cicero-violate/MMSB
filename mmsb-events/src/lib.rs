//! MMSB Events - Neutral Event Protocol
//! 
//! Defines event envelopes and EventSink trait.
//! NO runtime logic, NO authority.

pub mod events;
pub mod bus_traits;

pub use events::*;
pub use bus_traits::*;

/// EventSink - Abstract event emission
/// 
/// Semantic modules emit events via this trait.
/// Runtime (mmsb-service) implements this trait.
pub trait EventSink {
    fn emit(&self, event: AnyEvent);
}

/// Module lifecycle trait
/// 
/// Defines how modules participate in MMSB.
/// Modules attach to an EventSink, not to runtime internals.
pub trait Module: Send + Sync {
    /// Module name for identification
    fn name(&self) -> &str;
    
    /// Attach module to event sink
    fn attach(&mut self, sink: Box<dyn EventSink>);
    
    /// Detach module from event sink
    fn detach(&mut self);
}
