#![allow(dead_code)]
//! Action recommendations for dead code analysis.

use crate::dead_code_types::{ConfidenceLevel, DeadCodeCategory, RecommendedAction};

pub fn recommend_action(
    category: DeadCodeCategory,
    confidence: ConfidenceLevel,
    is_public_api: bool,
) -> RecommendedAction {
    match (category, confidence) {
        (DeadCodeCategory::Unreachable, ConfidenceLevel::CallGraph) if !is_public_api => {
            RecommendedAction::DeleteSafe
        }
        (DeadCodeCategory::Unreachable, _) => RecommendedAction::ManualReview,
        (DeadCodeCategory::ReachableUnused, _) => RecommendedAction::Quarantine,
        (DeadCodeCategory::TestOnly, _) => RecommendedAction::RelocateTests,
        (DeadCodeCategory::LatentPlanned, _) => RecommendedAction::Keep,
    }
}
