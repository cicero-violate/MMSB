//! MMSB Propagation
//!
//! Manages propagation of changes through the dependency graph.
//! 
//! Canonical dependencies:
//! - mmsb-primitives
//! - mmsb-physical  
//! - mmsb-allocator
//!
//! Authority: NONE (propagation engine only)

pub mod propagation_engine;
pub mod propagation_fastpath;
pub mod propagation_queue;
pub mod dag_propagation;
pub mod sparse_message_passing;
pub mod ring_buffer;
pub mod throughput_engine;
pub mod propagation_command_buffer;

pub use propagation_engine::PropagationEngine;
pub use propagation_queue::PropagationQueue;
