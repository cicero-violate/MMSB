#![allow(dead_code)]
//! Dead code category classification.

use crate::dead_code_call_graph::CallGraph;
use std::collections::HashSet;



pub fn is_reachable(symbol: &str, call_graph: &CallGraph, entrypoints: &HashSet<String>) -> bool {
    if entrypoints.is_empty() {
        return false;
    }
    let reachable = crate::dead_code_call_graph::compute_reachability(call_graph, entrypoints);
    reachable.contains(symbol)
}

