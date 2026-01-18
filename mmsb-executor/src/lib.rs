//! MMSB Executor Module
//! 
//! Execution substrate with ZERO authority.
//! Hardware management, scheduling, performance optimization.

pub mod module;
// pub mod physical;
// pub mod device;
// pub mod propagation;
// pub mod optimization;
pub mod allocation;

pub use module::ExecutorModule;
