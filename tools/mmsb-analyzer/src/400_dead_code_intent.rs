#![allow(dead_code)]
//! Intent detection aggregation for dead code classification.

pub use crate::dead_code_attribute_parser::detect_intent_signals;
use crate::dead_code_doc_comment_parser::item_name;
use crate::dead_code_types::{
    IntentMap, IntentMarker, IntentMetadata, IntentSource,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct DeadCodePolicy {
    pub planned_directories: Vec<PathBuf>,
    pub public_api_roots: Vec<PathBuf>,
    pub entrypoint_symbols: Vec<String>,
    pub treat_public_as_entrypoint: bool,
}



pub fn check_planned_directory(path: &Path, policy: Option<&DeadCodePolicy>) -> bool {
    let Some(policy) = policy else {
        return false;
    };
    for dir in &policy.planned_directories {
        if path.starts_with(dir) {
            return true;
        }
    }
    false
}

pub fn merge_intent_sources(
    attrs: IntentMap,
    docs: IntentMap,
    dir: IntentMap,
) -> IntentMap {
    let mut merged = IntentMap::new();
    for (symbol, items) in attrs {
        merged.entry(symbol).or_default().extend(items);
    }
    for (symbol, items) in docs {
        merged.entry(symbol).or_default().extend(items);
    }
    for (symbol, items) in dir {
        merged.entry(symbol).or_default().extend(items);
    }
    merged
}



pub(crate) fn planned_directory_intent(file: &Path, policy: Option<&DeadCodePolicy>) -> IntentMap {
    if !check_planned_directory(file, policy) {
        return IntentMap::new();
    }
    let mut map: IntentMap = HashMap::new();
    for symbol in collect_symbols(file) {
        map.entry(symbol).or_default().push(IntentMetadata {
            marker: IntentMarker::Planned,
            source: IntentSource::Directory,
            value: None,
        });
    }
    map
}

pub(crate) fn collect_symbols(file: &Path) -> Vec<String> {
    let contents = std::fs::read_to_string(file).unwrap_or_default();
    let parsed = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };
    parsed
        .items
        .iter()
        .filter_map(item_name)
        .collect()
}
