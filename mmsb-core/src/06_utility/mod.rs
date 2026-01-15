#![allow(unused_imports)]
//! Layer 6: Utility Engine
//!
//! Computes utility scores from telemetry to guide optimization decisions.
//!
//! # CRITICAL INVARIANT — UTILITY IS LEAF-ONLY
//!
//! ## Forbidden Imports
//!
//! **No code in `06_utility` may be imported by:**
//! - `01_page`
//! - `03_dag`
//! - `03_materialization`
//! - `04_propagation`
//! - `05_proof`
//!
//! ## Rationale
//!
//! Utility exists to **observe and analyze**, never to **execute or decide**.
//!
//! If any of the above phases depend on utility:
//! - Analysis becomes execution (breaks phase law)
//! - Observation mutates truth (breaks immutability)
//! - Measurement affects outcome (breaks determinism)
//!
//! ## Enforcement
//!
//! This boundary prevents:
//! ```text
//! ❌ propagation → utility → decision
//! ❌ proof → utility → validation
//! ❌ materialization → utility → state
//! ```
//!
//! Utility may only be used by:
//! ```text
//! ✅ 07_adaptive (proposals, not execution)
//! ✅ External tooling (monitoring, profiling)
//! ✅ Test harnesses (assertions, benchmarks)
//! ```
//!
//! **Violation of this invariant is a PHASE LAW VIOLATION.**

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
