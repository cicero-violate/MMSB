//! MMSB Events - Bus Protocol System

pub mod judgment_bus;
pub mod execution_bus;
pub mod state_bus;
pub mod learning_bus;
pub mod mmsb_subscription;

pub use judgment_bus::*;
pub use execution_bus::*;
pub use state_bus::*;
pub use learning_bus::*;
pub use mmsb_subscription::*;

/// EventSink - Legacy compatibility (deprecated)
pub trait EventSink {
    fn emit(&self, event: JudgmentApproved);
}

/// Module lifecycle trait
pub trait Module: Send + Sync {
    fn name(&self) -> &str;
    fn attach(&mut self, sink: Box<dyn EventSink>);
    fn detach(&mut self);
}
