#![allow(dead_code)]
//! Tier classification for predicted violations.

use crate::correction_plan_types::{ErrorTier, Severity, ViolationPrediction, ViolationType};

pub fn classify_tier(violation: &ViolationPrediction) -> ErrorTier {
    match (&violation.violation_type, &violation.severity) {
        (ViolationType::UnresolvedImport, _) => ErrorTier::Trivial,
        (ViolationType::BrokenReference, Severity::Low | Severity::Medium) => ErrorTier::Trivial,
        (ViolationType::NameCollision, _) => ErrorTier::Moderate,
        (ViolationType::LayerViolation, _) => ErrorTier::Moderate,
        (ViolationType::VisibilityMismatch, Severity::Low | Severity::Medium) => ErrorTier::Trivial,
        (ViolationType::VisibilityMismatch, Severity::High | Severity::Critical) => {
            ErrorTier::Moderate
        }
        (ViolationType::TypeMismatch, _) => ErrorTier::Complex,
        (ViolationType::OwnershipIssue, _) => ErrorTier::Complex,
        (ViolationType::BrokenReference, Severity::Critical | Severity::High) => ErrorTier::Complex,
    }
}
