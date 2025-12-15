//! Layer 6: Utility Engine
//!
//! Computes utility scores from telemetry to guide optimization decisions.

pub mod cpu_features;
pub mod telemetry;

pub use cpu_features::CpuFeatures;
pub use telemetry::{Telemetry, TelemetrySnapshot};
