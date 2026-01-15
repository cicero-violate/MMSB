#![allow(unused_imports)]
//! Layer 6: Utility Engine
//!
//! Computes utility scores from telemetry to guide optimization decisions.

pub mod cpu_features;
pub mod admission_proof;
pub mod delta_stream;
pub mod execution_proof;
pub mod structural_proof;
pub mod invariant_checker;
pub mod memory_monitor;
pub mod provenance_tracker;
pub mod telemetry;

pub use cpu_features::CpuFeatures;
pub use admission_proof::{
    build_admission_proof_streams, evaluate_admission, load_shell_policy,
    ADMISSION_PROOF_VERSION,
    MmsbAdmission, MmsbAdmissionProof, MmsbAdmissionProofStream,
    MmsbAdmissionProofError, PolicyDecision, PolicyError, ShellPolicy,
};
pub use delta_stream::{build_delta_streams, MmsbDelta, MmsbDeltaStream, MmsbDeltaStreamError};
pub use execution_proof::{
    build_execution_proof_stream, MmsbExecutionProof, MmsbExecutionProofStream,
    EXECUTION_PROOF_VERSION, MmsbExecutionProofError,
};
pub use structural_proof::{MmsbStructuralAdmissionProof, STRUCTURAL_PROOF_VERSION};
pub use invariant_checker::{EpochMonotonicity, GraphAcyclicity, Invariant, InvariantChecker, InvariantContext, InvariantResult, PageConsistency};
pub use crate::types::GCMetrics;
pub use memory_monitor::{MemoryMonitor, MemoryMonitorConfig, MemorySnapshot};
pub use provenance_tracker::{ProvenanceResult, ProvenanceTracker};
pub use telemetry::{Telemetry, TelemetrySnapshot};
