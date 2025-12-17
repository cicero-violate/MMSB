//! Layer 6: Utility Engine
//!
//! Computes utility scores from telemetry to guide optimization decisions.

pub mod cpu_features;
pub mod invariant_checker;
pub mod memory_monitor;
pub mod provenance_tracker;
pub mod telemetry;

pub use cpu_features::CpuFeatures;
pub use invariant_checker::{EpochMonotonicity, GraphAcyclicity, Invariant, InvariantChecker, InvariantContext, InvariantResult, PageConsistency};
pub use crate::types::GCMetrics;
pub use memory_monitor::{MemoryMonitor, MemoryMonitorConfig, MemorySnapshot};
pub use provenance_tracker::{ProvenanceResult, ProvenanceTracker};
pub use telemetry::{Telemetry, TelemetrySnapshot};
