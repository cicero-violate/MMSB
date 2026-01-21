//! MMSB Executor Module
//! 
//! Execution substrate with ZERO authority.
//! Hardware management, scheduling, performance optimization.

pub mod module;
pub mod execution_loop;
// pub mod physical;
// pub mod device;
// pub mod propagation;
// pub mod optimization;

pub use module::ExecutorModule;
pub use execution_loop::ExecutionLoop;
