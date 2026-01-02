//! Layer utility functions for layer-based dependency analysis
//! This module is at layer 010 to be accessible from all higher layers

use anyhow::Result;
use clap::Parser;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cluster_001::run_analysis;
use crate::cluster_010::gather_rust_files;
use crate::dead_code_actions::recommend_action;
use crate::dead_code_call_graph::{build_call_graph, classify_symbol, is_reachable};
use crate::dead_code_cli::{DeadCodeRunConfig, is_test_path, merge_intent_map, reason_for_category};
use crate::dead_code_confidence::{assign_confidence, Evidence};
use crate::dead_code_entrypoints::{collect_entrypoints, collect_exports, is_public_api};
use crate::dead_code_attribute_parser::{detect_test_modules, detect_test_symbols};
use crate::dead_code_intent::detect_intent_signals;
use crate::dead_code_report::{
    build_report, write_outputs, DeadCodeReportMetadata, DeadCodeReportWithMeta,
};
use crate::dead_code_test_boundaries::TestBoundaries;
use crate::dead_code_types::{DeadCodeCategory, DeadCodeItem};
use crate::types::{CodeElement, ElementType, Language, Visibility};

#[allow(unused_imports)]
pub use crate::cluster_001::{build_file_layers, detect_layer, gather_julia_files, julia_entry_paths};
#[allow(unused_imports)]
pub use crate::cluster_010::contains_tools;

/// Resolves the source root directory from a given root path
pub fn resolve_source_root(root: &Path) -> PathBuf {
    let src_candidate = root.join("src");
    if src_candidate.exists() && src_candidate.is_dir() {
        src_candidate
    } else {
        root.to_path_buf()
    }
}

/// Checks if a directory should be included in analysis
pub fn allow_analysis_dir(root: &Path, dir: &Path) -> bool {
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
    
    if name.starts_with('.') || name == "target" || name == "node_modules" {
        return false;
    }
    
    if let Ok(rel) = dir.strip_prefix(root) {
        if rel.components().any(|c| {
            let s = c.as_os_str().to_str().unwrap_or("");
            s.starts_with('.') || s == "target" || s == "node_modules"
        }) {
            return false;
        }
    }
    
    true
}



// ============================================================================
// CLI Entrypoint (from src/000_cluster_011.rs)
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "mmsb-analyzer")]
#[command(about = "MMSB Intelligence Substrate Analyzer", long_about = None)]
struct Args {
    /// Root directory to analyze
    #[arg(short, long, default_value = "../..")]
    root: PathBuf,

    /// Output directory for reports
    #[arg(short, long, default_value = "../../docs/analysis")]
    output: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip Julia file analysis
    #[arg(long)]
    skip_julia: bool,

    /// Run dead code analysis
    #[arg(long)]
    dead_code: bool,

    /// Filter dead code from downstream analysis
    #[arg(long)]
    dead_code_filter: bool,

    /// Output JSON dead code report
    #[arg(long)]
    dead_code_json: Option<PathBuf>,

    /// Output dead code summary markdown
    #[arg(long)]
    dead_code_summary: Option<PathBuf>,

    /// Dead code summary limit
    #[arg(long, default_value_t = 50)]
    dead_code_summary_limit: usize,

    /// Dead code policy file
    #[arg(long)]
    dead_code_policy: Option<PathBuf>,

    /// Generate correction intelligence JSON
    #[arg(long)]
    correction_intelligence: bool,

    /// Override correction intelligence JSON output path
    #[arg(long)]
    correction_json: Option<PathBuf>,

    /// Override verification policy JSON output path
    #[arg(long)]
    verification_policy_json: Option<PathBuf>,

    /// Generate path/module coherence slice (UpdatePath + rename-file actions)
    #[arg(long)]
    correction_path_slice: bool,

    /// Override path/module coherence slice output dir
    #[arg(long)]
    correction_path_slice_dir: Option<PathBuf>,

    /// Generate visibility slice (AdjustVisibility actions)
    #[arg(long)]
    correction_visibility_slice: bool,

    /// Override visibility slice output dir
    #[arg(long)]
    correction_visibility_slice_dir: Option<PathBuf>,

    /// Generate Phase-2 cluster slice (batch index, 1-based)
    #[arg(long)]
    correction_cluster_slice: Option<usize>,

    /// Override Phase-2 cluster slice output dir
    #[arg(long)]
    correction_cluster_slice_dir: Option<PathBuf>,

    /// Override Phase-2 cluster plan path
    #[arg(long)]
    correction_cluster_plan: Option<PathBuf>,
}

pub fn main() -> Result<()> {
    let args = Args::parse();

    let root_path = std::env::current_dir()?.join(&args.root).canonicalize()?;
    let output_path = std::env::current_dir()?
        .join(&args.output)
        .canonicalize()
        .unwrap_or_else(|_| {
            let p = std::env::current_dir().unwrap().join(&args.output);
            std::fs::create_dir_all(&p).ok();
            p.canonicalize().unwrap_or(p)
        });
    run_analysis(
        &root_path,
        &output_path,
        args.verbose,
        args.skip_julia,
        args.dead_code,
        args.dead_code_filter,
        args.dead_code_json,
        args.dead_code_summary,
        args.dead_code_summary_limit,
        args.dead_code_policy,
        args.correction_intelligence,
        args.correction_json,
        args.verification_policy_json,
        args.correction_path_slice,
        args.correction_path_slice_dir,
        args.correction_visibility_slice,
        args.correction_visibility_slice_dir,
        args.correction_cluster_slice,
        args.correction_cluster_slice_dir,
        args.correction_cluster_plan,
    )
}

pub fn run_dead_code_pipeline(
    elements: &[CodeElement],
    config: &DeadCodeRunConfig,
) -> Result<DeadCodeReportWithMeta> {
    let rust_files = gather_rust_files(&config.root);
    let mut intent_map = HashMap::new();
    let mut test_boundaries = TestBoundaries::default();

    for file in &rust_files {
        let intents = detect_intent_signals(file, config.policy.as_ref());
        merge_intent_map(&mut intent_map, intents);
        let test_modules = detect_test_modules(file);
        test_boundaries.test_modules.extend(test_modules);
        let test_symbols = detect_test_symbols(file);
        test_boundaries.test_symbols.extend(test_symbols);
        if is_test_path(file) {
            test_boundaries.test_files.insert(file.clone());
        }
    }

    let call_graph = build_call_graph(elements);
    let entrypoints = collect_entrypoints(elements, config.policy.as_ref());
    let exports = collect_exports(&config.root);

    let mut items = Vec::new();
    for element in elements {
        if element.element_type != ElementType::Function {
            continue;
        }
        if element.language != Language::Rust {
            continue;
        }
        let category = classify_symbol(
            &element.name,
            &call_graph,
            &intent_map,
            &test_boundaries,
            &entrypoints,
            config.policy.as_ref(),
        );
        let intent_tag = intent_map.contains_key(&element.name);
        let test_reference = test_boundaries.test_symbols.contains(&element.name);
        let call_graph_proven =
            category == DeadCodeCategory::Unreachable && is_reachable(&element.name, &call_graph, &entrypoints) == false;

        let mut item = DeadCodeItem {
            symbol: element.name.clone(),
            file: PathBuf::from(&element.file_path),
            line: element.line_number,
            category,
            confidence: crate::dead_code_types::ConfidenceLevel::Heuristic,
            action: crate::dead_code_types::RecommendedAction::ManualReview,
            reason: reason_for_category(category, intent_tag, test_reference),
        };

        let confidence = assign_confidence(
            &item,
            &Evidence {
                intent_tag,
                test_reference,
                call_graph_proven,
            },
        );
        item.confidence = confidence;

        let public_api = is_public_api(&element.name, &exports)
            || matches!(element.visibility, Visibility::Public);
        item.action = recommend_action(category, confidence, public_api);

        items.push(item);
    }

    let metadata = DeadCodeReportMetadata {
        analyzer_version: env!("CARGO_PKG_VERSION").to_string(),
        project_root: config.root.display().to_string(),
        entrypoints_found: entrypoints.len(),
    };
    let report = build_report(
        chrono::Local::now().to_rfc3339(),
        items,
        metadata,
    );

    write_outputs(&report, config)?;
    Ok(report)
}
