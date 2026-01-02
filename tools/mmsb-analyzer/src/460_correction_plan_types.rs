#![allow(dead_code)]
//! Correction intelligence core types.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::types::Visibility;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorTier {
    Trivial,
    Moderate,
    Complex,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CorrectionStrategy {
    AddImport { module_path: String, symbol: String },
    UpdatePath { old_path: String, new_path: String },
    ReExport { from_module: String, symbol: String },
    RenameWithSuffix { original: String, suffix: String },
    MoveToLayer { function: String, target_layer: String },
    EnsureImports { function: String, target_layer: String },
    UpdateCaller { caller_file: PathBuf, old_ref: String, new_ref: String },
    AdjustVisibility {
        symbol: String,
        file: PathBuf,
        from: Visibility,
        to: Visibility,
        reason: String,
    },
    VisibilityPlan {
        symbol: String,
        file: PathBuf,
        current: Visibility,
        default_policy: String,
        options: Vec<VisibilityPlanOption>,
        notes: String,
    },
    ManualReview { reason: String, context: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisibilityPlanOption {
    pub policy: String,
    pub target: Visibility,
    pub requires_consent: bool,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionPlan {
    pub action_id: String,
    pub tier: ErrorTier,
    pub predicted_violations: Vec<ViolationPrediction>,
    pub strategies: Vec<CorrectionStrategy>,
    pub confidence: f64,
    pub estimated_fix_time_seconds: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ViolationPrediction {
    pub violation_type: ViolationType,
    pub affected_files: Vec<PathBuf>,
    pub severity: Severity,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ViolationType {
    UnresolvedImport,
    BrokenReference,
    NameCollision,
    LayerViolation,
    VisibilityMismatch,
    TypeMismatch,
    OwnershipIssue,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RefactorAction {
    MoveFunction {
        function: String,
        from: PathBuf,
        to: PathBuf,
        required_layer: Option<String>,
    },
    RenameFunction {
        old_name: String,
        new_name: String,
        file: PathBuf,
    },
    RenameFile {
        from: PathBuf,
        to: PathBuf,
    },
    CreateFile {
        path: PathBuf,
    },
    AdjustVisibility {
        symbol: String,
        file: PathBuf,
        from: Visibility,
        to: Visibility,
        reason: String,
    },
}

impl RefactorAction {
    pub fn action_id(&self) -> String {
        match self {
            RefactorAction::MoveFunction { function, to, .. } => {
                format!("move_{}_to_{}", function, to.display())
            }
            RefactorAction::RenameFunction { old_name, new_name, .. } => {
                format!("rename_{}_to_{}", old_name, new_name)
            }
            RefactorAction::RenameFile { from, to } => {
                format!("rename_file_{}_to_{}", from.display(), to.display())
            }
            RefactorAction::CreateFile { path } => format!("create_file_{}", path.display()),
            RefactorAction::AdjustVisibility { symbol, file, .. } => {
                format!("adjust_visibility_{}_in_{}", symbol, file.display())
            }
        }
    }
}
