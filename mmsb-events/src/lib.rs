//! MMSB Events - Bus Protocol System

pub mod judgment_bus;
pub mod execution_bus;
pub mod state_bus;
pub mod learning_bus;
pub mod memory_reader;
// pub mod mmsb_subscription;

pub use judgment_bus::*;
pub use execution_bus::*;
pub use state_bus::*;
pub use learning_bus::*;
pub use memory_reader::*;
// pub use mmsb_subscription::*;

/// Module lifecycle trait
pub trait Module: Send + Sync {
    fn name(&self) -> &str;
    fn attach_judgment_bus(&mut self, bus: Box<dyn JudgmentBus>);
    fn detach(&mut self);
}
