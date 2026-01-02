#![allow(dead_code)]
//! Filter dead code from downstream analysis inputs.

use crate::dead_code_types::{DeadCodeCategory};
use crate::dead_code_report::DeadCodeReportWithMeta;
use crate::types::CodeElement;
use std::collections::HashSet;

pub fn filter_dead_code_elements(
    elements: &[CodeElement],
    report: &DeadCodeReportWithMeta,
) -> Vec<CodeElement> {
    let excluded = collect_excluded_symbols(report);
    elements
        .iter()
        .filter(|el| !excluded.contains(&el.name))
        .cloned()
        .collect()
}

pub fn should_exclude_from_analysis(category: DeadCodeCategory) -> bool {
    matches!(
        category,
        DeadCodeCategory::Unreachable | DeadCodeCategory::TestOnly
    )
}

fn collect_excluded_symbols(report: &DeadCodeReportWithMeta) -> HashSet<String> {
    report
        .items
        .iter()
        .filter(|item| should_exclude_from_analysis(item.category))
        .map(|item| item.symbol.clone())
        .collect()
}
