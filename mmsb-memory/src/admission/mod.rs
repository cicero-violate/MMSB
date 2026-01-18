// Admission gateway - verifies JudgmentProof and epoch validity

pub mod admission_gate;
pub mod replay_protection;

pub use admission_gate::{AdmissionGate, AdmissionResult, AdmissionRejection};
pub use replay_protection::{ReplayProtection, ReplayError};
