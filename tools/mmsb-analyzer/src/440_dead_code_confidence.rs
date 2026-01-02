#![allow(dead_code)]
//! Confidence scoring for dead code analysis.

use crate::dead_code_types::{ConfidenceLevel, DeadCodeCategory, DeadCodeItem};

#[derive(Debug, Clone, Default)]
pub struct Evidence {
    pub intent_tag: bool,
    pub test_reference: bool,
    pub call_graph_proven: bool,
}

pub fn assign_confidence(item: &DeadCodeItem, evidence: &Evidence) -> ConfidenceLevel {
    if evidence.intent_tag || matches!(item.category, DeadCodeCategory::LatentPlanned) {
        return ConfidenceLevel::IntentTag;
    }
    if evidence.test_reference || matches!(item.category, DeadCodeCategory::TestOnly) {
        return ConfidenceLevel::TestReference;
    }
    if evidence.call_graph_proven || matches!(item.category, DeadCodeCategory::Unreachable) {
        return ConfidenceLevel::CallGraph;
    }
    ConfidenceLevel::Heuristic
}
