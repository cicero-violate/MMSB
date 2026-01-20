// src/lib.rs
//! MMSB Service Runtime
//!
//! Process-local wiring only.
//! No authority. No truth. No replay.
//!
//! All correctness flows through MMSB memory.

mod runtime;
pub use runtime::Runtime;
mod protocol_signal_out;
mod protocol_signal_in;
mod scheduler;

pub use protocol_signal_out::ProtocolSignalOut;
pub use protocol_signal_in::ProtocolSignalIn;
pub use scheduler::Scheduler;

// src/lib.rs (additions)
mod service;
mod service_context;
mod wake_source;
pub mod services;

pub use service::Service;
pub use service_context::ServiceContext;
pub use wake_source::WakeSource;
