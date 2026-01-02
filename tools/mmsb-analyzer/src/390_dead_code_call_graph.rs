#![allow(dead_code)]
//! Dead code call graph utilities.

use crate::dead_code_test_boundaries::TestBoundaries;
use crate::types::{CodeElement, ElementType, Language};
use std::collections::{HashMap, HashSet, VecDeque};

pub type CallGraph = HashMap<String, Vec<String>>;

pub fn build_call_graph(elements: &[CodeElement]) -> CallGraph {
    let mut graph: CallGraph = HashMap::new();
    for element in elements {
        if element.element_type != ElementType::Function {
            continue;
        }
        if element.language != Language::Rust {
            continue;
        }
        let entry = graph.entry(element.name.clone()).or_default();
        entry.extend(element.calls.iter().cloned());
    }
    graph
}

pub fn build_reverse_call_graph(graph: &CallGraph) -> CallGraph {
    let mut reverse: CallGraph = HashMap::new();
    for (caller, callees) in graph {
        for callee in callees {
            reverse.entry(callee.clone()).or_default().push(caller.clone());
        }
    }
    reverse
}

pub fn compute_reachability(graph: &CallGraph, entrypoints: &HashSet<String>) -> HashSet<String> {
    let mut reachable = HashSet::new();
    let mut queue: VecDeque<String> = entrypoints.iter().cloned().collect();

    while let Some(node) = queue.pop_front() {
        if !reachable.insert(node.clone()) {
            continue;
        }
        if let Some(callees) = graph.get(&node) {
            for callee in callees {
                if !reachable.contains(callee) {
                    queue.push_back(callee.clone());
                }
            }
        }
    }

    reachable
}

pub fn is_reachable(
    symbol: &str,
    graph: &CallGraph,
    entrypoints: &HashSet<String>,
) -> bool {
    if entrypoints.is_empty() {
        return false;
    }
    compute_reachability(graph, entrypoints).contains(symbol)
}

#[path = "040_classify_symbol.rs"]
mod moved_classify_symbol;
pub use moved_classify_symbol::classify_symbol;

pub fn is_test_only(
    symbol: &str,
    call_graph: &CallGraph,
    test_boundaries: &TestBoundaries,
) -> bool {
    if test_boundaries.test_symbols.contains(symbol) {
        return true;
    }
    let reverse = build_reverse_call_graph(call_graph);
    let callers = reverse.get(symbol);
    let Some(callers) = callers else {
        return false;
    };
    if callers.is_empty() {
        return false;
    }
    callers
        .iter()
        .all(|caller| test_boundaries.test_symbols.contains(caller))
}
