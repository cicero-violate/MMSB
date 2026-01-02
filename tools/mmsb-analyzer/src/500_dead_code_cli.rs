#![allow(dead_code)]
//! Dead code analysis pipeline runner.

use crate::dead_code_intent::DeadCodePolicy;
use crate::dead_code_types::DeadCodeCategory;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DeadCodeRunConfig {
    pub root: PathBuf,
    pub output_dir: PathBuf,
    pub policy: Option<DeadCodePolicy>,
    pub write_json: Option<PathBuf>,
    pub write_summary: Option<PathBuf>,
    pub summary_limit: usize,
}

pub use crate::layer_utilities::run_dead_code_pipeline;





pub(crate) fn merge_intent_map(base: &mut HashMap<String, Vec<crate::dead_code_types::IntentMetadata>>, next: HashMap<String, Vec<crate::dead_code_types::IntentMetadata>>) {
    for (symbol, items) in next {
        base.entry(symbol).or_default().extend(items);
    }
}

pub(crate) fn reason_for_category(category: DeadCodeCategory, intent_tag: bool, test_reference: bool) -> String {
    match category {
        DeadCodeCategory::LatentPlanned => {
            if intent_tag {
                "Intent tag present".to_string()
            } else {
                "Intent directory policy".to_string()
            }
        }
        DeadCodeCategory::TestOnly => {
            if test_reference {
                "Called only by test symbols".to_string()
            } else {
                "Defined in test-only module".to_string()
            }
        }
        DeadCodeCategory::Unreachable => "No callers reachable from entrypoints".to_string(),
        DeadCodeCategory::ReachableUnused => "Reachable but unused in execution".to_string(),
    }
}

pub(crate) fn is_test_path(path: &Path) -> bool {
    path.components().any(|c| {
        let name = c.as_os_str().to_str().unwrap_or("");
        name == "tests" || name == "test" || name == "benches"
    })
}
