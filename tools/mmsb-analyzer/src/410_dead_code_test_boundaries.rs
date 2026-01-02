#![allow(dead_code)]
//! Test boundary detection for dead code classification.

use crate::dead_code_call_graph::{build_reverse_call_graph, CallGraph};
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use syn::{Attribute, Item};

#[derive(Debug, Clone, Default)]
pub struct TestBoundaries {
    pub test_modules: HashSet<String>,
    pub test_symbols: HashSet<String>,
    pub test_files: HashSet<PathBuf>,
}







pub fn find_test_callers(
    symbol: &str,
    call_graph: &CallGraph,
    test_symbols: &HashSet<String>,
) -> Vec<String> {
    if test_symbols.is_empty() {
        return Vec::new();
    }
    let reverse = build_reverse_call_graph(call_graph);
    let mut callers = Vec::new();
    let mut visited = HashSet::new();
    let mut queue: VecDeque<String> = reverse
        .get(symbol)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();

    while let Some(caller) = queue.pop_front() {
        if !visited.insert(caller.clone()) {
            continue;
        }
        if test_symbols.contains(&caller) {
            callers.push(caller.clone());
        }
        if let Some(next) = reverse.get(&caller) {
            for parent in next {
                if !visited.contains(parent) {
                    queue.push_back(parent.clone());
                }
            }
        }
    }

    callers
}

pub(crate) fn has_test_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let path = attr.path();
        if path.is_ident("test") {
            return true;
        }
        let last = path.segments.last().map(|seg| seg.ident.to_string());
        matches!(last.as_deref(), Some("test"))
    })
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Fn(item_fn) => &item_fn.attrs,
        Item::Struct(item_struct) => &item_struct.attrs,
        Item::Enum(item_enum) => &item_enum.attrs,
        Item::Mod(item_mod) => &item_mod.attrs,
        Item::Trait(item_trait) => &item_trait.attrs,
        _ => &[],
    }
}
