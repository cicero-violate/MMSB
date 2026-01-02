#![allow(dead_code)]
//! Confidence scoring for predicted violations.

use crate::correction_plan_types::{ViolationPrediction, ViolationType};

#[derive(Debug, Clone, Default)]
pub struct PredictionContext {
    pub has_test_coverage: bool,
}

pub fn compute_confidence(
    prediction: &ViolationPrediction,
    context: &PredictionContext,
) -> f64 {
    let base: f64 = match prediction.violation_type {
        ViolationType::UnresolvedImport => 0.95,
        ViolationType::NameCollision => 1.0,
        ViolationType::LayerViolation => 1.0,
        ViolationType::VisibilityMismatch => 0.8,
        ViolationType::BrokenReference => 0.85,
        ViolationType::TypeMismatch => 0.6,
        ViolationType::OwnershipIssue => 0.5,
    };
    let multiplier: f64 = if context.has_test_coverage { 1.1 } else { 0.9 };
    (base * multiplier).min(1.0)
}
