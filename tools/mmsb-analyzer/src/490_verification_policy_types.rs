#![allow(dead_code)]
//! Verification policy types for correction intelligence.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationPolicy {
    pub action_id: String,
    pub scope: VerificationScope,
    pub required_checks: Vec<VerificationCheck>,
    pub incremental_eligible: bool,
    pub estimated_time_seconds: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerificationScope {
    SyntaxOnly { files: Vec<PathBuf> },
    ModuleLocal { module: String, transitive_depth: u32 },
    CallerChain { root_function: String },
    FullWorkspace,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerificationCheck {
    CargoCheck,
    CargoTest { filter: Option<String> },
    InvariantValidation { invariant_ids: Vec<String> },
    QualityMetrics { thresholds: QualityThresholds },
    ManualInspection { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_cohesion_delta: f64,
    pub max_violation_delta: i32,
    pub max_complexity_delta: f64,
}
