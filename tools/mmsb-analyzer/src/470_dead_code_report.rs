#![allow(dead_code)]
//! JSON report generator for dead code analysis.

use crate::dead_code_cli::DeadCodeRunConfig;
use crate::dead_code_report_split::{write_plan_markdown, write_summary_markdown};
use crate::dead_code_types::{DeadCodeItem, DeadCodeReport, DeadCodeSummary};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeReportMetadata {
    pub analyzer_version: String,
    pub project_root: String,
    pub entrypoints_found: usize,
}

pub fn build_report(
    timestamp: String,
    items: Vec<DeadCodeItem>,
    metadata: DeadCodeReportMetadata,
) -> DeadCodeReportWithMeta {
    let summary = DeadCodeSummary::from_items(&items);
    DeadCodeReportWithMeta {
        timestamp,
        summary,
        items,
        metadata,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeReportWithMeta {
    pub timestamp: String,
    pub summary: DeadCodeSummary,
    pub items: Vec<DeadCodeItem>,
    pub metadata: DeadCodeReportMetadata,
}

pub fn write_report(path: &Path, report: &DeadCodeReportWithMeta) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(path, json)
}

pub fn build_basic_report(timestamp: String, items: Vec<DeadCodeItem>) -> DeadCodeReport {
    let summary = DeadCodeSummary::from_items(&items);
    DeadCodeReport {
        timestamp,
        summary,
        items,
    }
}

pub(crate) fn write_outputs(report: &DeadCodeReportWithMeta, config: &DeadCodeRunConfig) -> Result<()> {
    let json_path = config
        .write_json
        .clone()
        .unwrap_or_else(|| config.output_dir.join("dead_code_full.json"));
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_report(&json_path, report)?;

    let summary_path = config
        .write_summary
        .clone()
        .unwrap_or_else(|| config.output_dir.join("dead_code_summary.md"));
    if let Some(parent) = summary_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_summary_markdown(&summary_path, report, config.summary_limit)?;

    let plans_dir = summary_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| config.output_dir.clone());
    let plans_path = plans_dir.join("dead_code_plans.md");
    write_plan_markdown(&plans_path, report, config.summary_limit)?;
    Ok(())
}
