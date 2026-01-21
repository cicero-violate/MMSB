//! MMSB Chromium - Chrome DevTools Protocol (CDP) Integration
//!
//! Handles bidirectional communication with Chrome:
//! - Incoming: LLM responses from ChatGPT → IntentBus
//! - Outgoing: Execution results → Chrome UI
//!
//! Authority: NONE (pure I/O adapter)

pub mod connection;
pub mod incoming;
pub mod outgoing;

pub use connection::{ChromeConnection, ChromeError};
pub use incoming::{CapturedMessage, MessageCapture};
pub use outgoing::MessageSender;
