//! MMSB Executor Module
//! 
//! Mechanical-only execution module with ZERO authority.
//! Consumes JudgmentApproved events and prepares ExecutionRequests.

pub mod module;

pub use module::ExecutorModule;
