mod control_flow;
mod dependency;
mod julia_parser;
mod report;
mod rust_parser;
mod types;

use anyhow::{Context, Result};
use clap::Parser;
use std::env;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::control_flow::ControlFlowAnalyzer;
use crate::dependency::{
    order_julia_files_by_dependency, order_rust_files_by_dependency, LayerGraph,
};
use crate::julia_parser::JuliaAnalyzer;
use crate::report::ReportGenerator;
use crate::rust_parser::RustAnalyzer;
use crate::types::AnalysisResult;

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

    /// Path to Julia analyzer script
    #[arg(short, long, default_value = "./src/00_main.jl")]
    julia_script: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip Julia file analysis
    #[arg(long)]
    skip_julia: bool,
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("Usage: {} <project-root>", args[0]);
        std::process::exit(1);
    }

    let project_root = PathBuf::from(&args[1]);
    let output_path = project_root.join("analysis_report");
    fs::create_dir_all(&output_path)?;

    println!("Starting MMSB analysis on: {}", project_root.display());

    // === 1. Discover and order files by layer ===
    println!("\nDiscovering and ordering source files...");
    let rust_files = gather_rust_files(&project_root)?;
    let julia_files = gather_julia_files(&project_root)?;

    let rust_layer_graph = build_layer_dependency_graph(&rust_files);
    let julia_layer_graph = build_layer_dependency_graph(&julia_files);

    let rust_ordered = topological_sort_files(&rust_files, &rust_layer_graph);
    let julia_ordered = topological_sort_files(&julia_files, &julia_layer_graph);

    // === 2. Analyze Rust files ===
    println!("\nAnalyzing {} Rust files...", rust_ordered.len());
    let rust_analyzer = RustAnalyzer::new();
    let mut combined_result = AnalysisResult::new();

    for file in rust_ordered {
        if let Ok(result) = rust_analyzer.analyze_file(&file) {
            combined_result.merge(result);
        }
    }

    // === 3. Analyze Julia files ===
    println!("\nAnalyzing {} Julia files...", julia_ordered.len());
    let julia_analyzer = JuliaAnalyzer::new();
    for file in julia_ordered {
        if let Ok(result) = julia_analyzer.analyze_file(&file) {
            combined_result.merge(result);
        }
    }

    // === 4. Build call graph ===
    println!("\nBuilding control flow graph...");
    let mut cf_analyzer = ControlFlowAnalyzer::new();
    cf_analyzer.build_call_graph(&combined_result);

    // === 5. EXPORT RICH PROGRAM-LEVEL CFG TO DOT (the big new feature) ===
    println!("\nExporting complete program CFG (with clusters, ENTRY/EXIT, clickable nodes)...");
    use crate::types::{FunctionCfg, ProgramCFG};
    use std::collections::HashMap;

    let mut program_cfg = ProgramCFG {
        functions: HashMap::new(),
        call_edges: Vec::new(),
    };

    // Insert all CFGs (Rust + Julia if any)
    for cfg in &combined_result.cfgs {
        program_cfg
            .functions
            .insert(cfg.function.clone(), cfg.clone());
    }

    // Extract inter-function call edges from petgraph
    for edge in cf_analyzer.graph.edge_indices() {
        let (source, target) = cf_analyzer.graph.edge_endpoints(edge).unwrap();
        let caller_node = &cf_analyzer.graph[source];
        let callee_node = &cf_analyzer.graph[target];

        // Strip module path, keep just function name (assumes uniqueness or you can keep full path)
        let caller = caller_node.split("::").last().unwrap_or(caller_node).to_string();
        let callee = callee_node.split("::").last().unwrap_or(callee_node).to_string();

        if !caller.is_empty() && !callee.is_empty() {
            program_cfg.call_edges.push((caller, callee));
        }
    }

    // Ensure output directory
    let cfg_dir = output_path.join("cfg");
    fs::create_dir_all(&cfg_dir)?;

    let dot_path = cfg_dir.join("complete_program.dot");
    crate::dot_exporter::export_complete_program_dot(&program_cfg, dot_path.to_str().unwrap())?;
    println!("Program CFG exported to: {}", dot_path.display());

    // Optional: auto-generate PNG
    #[cfg(feature = "png")]
    {
        let png_path = cfg_dir.join("complete_program.png");
        if let Ok(dot_path_str) = dot_path.to_str() {
            if let Ok(png_path_str) = png_path.to_str() {
                let status = std::process::Command::new("dot")
                    .args(&["-Tpng", dot_path_str, "-o", png_path_str])
                    .status();
                if status.map_or(false, |s| s.success()) {
                    println!("PNG rendered: {}", png_path.display());
                }
            }
        }
    }

    // === 6. Generate final reports ===
    println!("\nGenerating Markdown reports...");
    let report_gen = ReportGenerator::new(output_path.to_string_lossy().to_string());
    report_gen
        .generate_all(
            &combined_result,
            &cf_analyzer,
            &rust_layer_graph,
            &julia_layer_graph,
        )
        .context("Failed to generate reports")?;

    println!("\nAnalysis complete!");
    println!("Report: {}/index.html", output_path.display());
    println!("Interactive CFG: {}/cfg/complete_program.dot (open with xdot!)", output_path.display());

    Ok(())
}

fn gather_rust_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
        .filter(|e| {
            !e.path().components().any(|c| {
                c.as_os_str() == "target" || c.as_os_str() == "examples" || c.as_os_str() == "tools"
            })
        })
        .map(|entry| entry.into_path())
        .collect()
}

fn gather_julia_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "jl"))
        .filter(|e| {
            !e.path().components().any(|c| {
                let name = c.as_os_str();
                name == "test" || name == "examples" || name == "tools" || name == ".julia"
            })
        })
        .map(|entry| entry.into_path())
        .collect()
}
