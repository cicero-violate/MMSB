#![allow(dead_code)]
//! Dead code report split utilities.

use crate::dead_code_report::DeadCodeReportWithMeta;
use crate::dead_code_types::{DeadCodeCategory, DeadCodeItem, RecommendedAction};
use std::path::Path;

pub fn write_summary_markdown(
    path: &Path,
    report: &DeadCodeReportWithMeta,
    limit: usize,
) -> std::io::Result<()> {
    let mut content = String::new();
    content.push_str("# Dead Code Summary\n\n");
    content.push_str(&format!(
        "Generated: {}\n\n",
        report.timestamp
    ));
    content.push_str("## Summary Counts\n\n");
    content.push_str(&format!(
        "- Unreachable: {}\n- Reachable-unused: {}\n- Test-only: {}\n- Latent/planned: {}\n- Total analyzed: {}\n\n",
        report.summary.unreachable,
        report.summary.reachable_unused,
        report.summary.test_only,
        report.summary.latent_planned,
        report.summary.total_analyzed
    ));

    let items = top_items(&report.items, limit);
    content.push_str("## Top Findings\n\n");
    if items.is_empty() {
        content.push_str("- None.\n");
    } else {
        for item in items {
            content.push_str(&format!(
                "- `{}` in `{}` — {:?} / {:?} / {:?}\n",
                item.symbol,
                item.file.display(),
                item.category,
                item.confidence,
                item.action
            ));
        }
    }
    content.push('\n');

    std::fs::write(path, content)
}

pub fn write_plan_markdown(
    path: &Path,
    report: &DeadCodeReportWithMeta,
    limit: usize,
) -> std::io::Result<()> {
    let mut content = String::new();
    content.push_str("# Dead Code Plans (Review Only)\n\n");
    content.push_str(&format!(
        "Generated: {}\n\n",
        report.timestamp
    ));
    content.push_str("Policy: review_only. No automatic deletion or moves.\n");
    content.push_str("Guards: never delete public API; delete_safe requires manual confirmation + compiler dead_code warnings.\n\n");

    let items = top_items(&report.items, limit);
    content.push_str("## Planned Items\n\n");
    if items.is_empty() {
        content.push_str("- None.\n");
    } else {
        for item in items {
            let plan = plan_options(&item);
            content.push_str(&format!(
                "- `{}` in `{}` — {:?} / {:?} / {:?}\n  Plan: {}\n",
                item.symbol,
                item.file.display(),
                item.category,
                item.confidence,
                item.action,
                plan
            ));
        }
    }
    content.push('\n');

    std::fs::write(path, content)
}

fn top_items(items: &[DeadCodeItem], limit: usize) -> Vec<DeadCodeItem> {
    let mut items = items.to_vec();
    items.sort_by_key(|item| item.action as u8);
    if items.len() > limit {
        items.truncate(limit);
    }
    items
}

fn plan_options(item: &DeadCodeItem) -> String {
    let options = match item.category {
        DeadCodeCategory::Unreachable => "keep | quarantine | delete_safe (manual confirm)",
        DeadCodeCategory::ReachableUnused => "keep | quarantine | annotate_intent",
        DeadCodeCategory::TestOnly => "relocate_tests | keep",
        DeadCodeCategory::LatentPlanned => "keep | annotate_intent",
    };
    let mut plan = format!("review_only; options: {}", options);
    if item.action == RecommendedAction::DeleteSafe {
        plan.push_str("; requires dead_code warning");
    }
    plan
}
