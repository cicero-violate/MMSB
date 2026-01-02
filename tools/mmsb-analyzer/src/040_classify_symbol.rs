use crate::dead_code_call_graph::{is_reachable, is_test_only, CallGraph};
use crate::dead_code_intent::DeadCodePolicy;
use crate::dead_code_test_boundaries::TestBoundaries;
use crate::dead_code_types::{DeadCodeCategory, IntentMap};
use std::collections::HashSet;

pub fn classify_symbol(
    symbol: &str,
    call_graph: &CallGraph,
    intent_map: &IntentMap,
    test_boundaries: &TestBoundaries,
    entrypoints: &HashSet<String>,
    _policy: Option<&DeadCodePolicy>,
) -> DeadCodeCategory {
    if intent_map.contains_key(symbol) {
        return DeadCodeCategory::LatentPlanned;
    }

    if is_test_only(symbol, call_graph, test_boundaries) {
        return DeadCodeCategory::TestOnly;
    }

    if !is_reachable(symbol, call_graph, entrypoints) {
        return DeadCodeCategory::Unreachable;
    }

    DeadCodeCategory::ReachableUnused
}
