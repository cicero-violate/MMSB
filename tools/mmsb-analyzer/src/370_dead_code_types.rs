#![allow(dead_code)]
//! Dead code analysis types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadCodeCategory {
    Unreachable,
    ReachableUnused,
    TestOnly,
    LatentPlanned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    CallGraph,
    TestReference,
    IntentTag,
    Heuristic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    DeleteSafe,
    Quarantine,
    RelocateTests,
    Keep,
    ManualReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeItem {
    pub symbol: String,
    pub file: PathBuf,
    pub line: usize,
    pub category: DeadCodeCategory,
    pub confidence: ConfidenceLevel,
    pub action: RecommendedAction,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeadCodeSummary {
    pub unreachable: usize,
    pub reachable_unused: usize,
    pub test_only: usize,
    pub latent_planned: usize,
    pub total_analyzed: usize,
}

impl DeadCodeSummary {
    pub fn from_items(items: &[DeadCodeItem]) -> Self {
        let mut summary = DeadCodeSummary::default();
        summary.total_analyzed = items.len();
        for item in items {
            match item.category {
                DeadCodeCategory::Unreachable => summary.unreachable += 1,
                DeadCodeCategory::ReachableUnused => summary.reachable_unused += 1,
                DeadCodeCategory::TestOnly => summary.test_only += 1,
                DeadCodeCategory::LatentPlanned => summary.latent_planned += 1,
            }
        }
        summary
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeReport {
    pub timestamp: String,
    pub summary: DeadCodeSummary,
    pub items: Vec<DeadCodeItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntentSource {
    Attribute,
    DocComment,
    Directory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntentMarker {
    Latent,
    Planned,
    Future,
    DeprecatedPlanned,
}

impl IntentMarker {
    pub fn from_comment(comment: &str) -> Option<Self> {
        let lower = comment.to_ascii_lowercase();
        if lower.contains("@deprecated-planned") {
            return Some(IntentMarker::DeprecatedPlanned);
        }
        if lower.contains("@planned") {
            return Some(IntentMarker::Planned);
        }
        if lower.contains("@future") {
            return Some(IntentMarker::Future);
        }
        if lower.contains("@latent") {
            return Some(IntentMarker::Latent);
        }
        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMetadata {
    pub marker: IntentMarker,
    pub source: IntentSource,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentTag {
    pub symbol: String,
    pub file: PathBuf,
    pub line: Option<usize>,
    pub marker: IntentMarker,
    pub source: IntentSource,
    pub value: Option<String>,
}

pub type IntentMap = HashMap<String, Vec<IntentMetadata>>;
