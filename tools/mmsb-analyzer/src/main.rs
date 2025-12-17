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
use crate::dependency::{order_julia_files_by_dependency, order_rust_files_by_dependency};
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
    #[arg(short, long, default_value = "./src/analyzer.jl")]
    julia_script: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Resolve to absolute paths
    let root_path = env::current_dir()?.join(&args.root).canonicalize()?;
    let output_path = env::current_dir()?
        .join(&args.output)
        .canonicalize()
        .unwrap_or_else(|_| {
            let p = env::current_dir().unwrap().join(&args.output);
            std::fs::create_dir_all(&p).ok();
            p.canonicalize().unwrap_or(p)
        });
    let julia_script_path = env::current_dir()?
        .join(&args.julia_script)
        .canonicalize()?;

    println!("MMSB Intelligence Substrate Analyzer");
    println!("=====================================\n");
    println!("Root directory: {:?}", root_path);
    println!("Output directory: {:?}", output_path);
    println!("Julia script: {:?}\n", julia_script_path);

    let rust_analyzer = RustAnalyzer::new(root_path.to_string_lossy().to_string());

    let julia_analyzer = JuliaAnalyzer::new(
        root_path.clone(),
        julia_script_path.clone(),
        output_path.join("cfg/dots"),
    );

    let mut combined_result = AnalysisResult::new();

    // Scan for Rust files
    println!("Scanning Rust files (dependency-ordered)...");
    let mut rust_count = 0;
    let rust_files = gather_rust_files(&root_path);
    let (ordered_rust_files, rust_layer_graph) =
        order_rust_files_by_dependency(&rust_files, &root_path)
            .context("Failed to resolve Rust dependency order")?;

    for path in ordered_rust_files {
        if args.verbose {
            println!("  Analyzing: {:?}", path);
        }

        match rust_analyzer.analyze_file(&path) {
            Ok(result) => {
                rust_count += 1;
                combined_result.merge(result);
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze {:?}: {}", path, e);
            }
        }
    }
    println!("Analyzed {} Rust files", rust_count);

    // Scan for Julia files
    println!("\nScanning Julia files (dependency-ordered)...");
    let mut julia_count = 0;
    let julia_files = gather_julia_files(&root_path);
    let (ordered_julia_files, julia_layer_graph) =
        order_julia_files_by_dependency(&julia_files, &root_path)
            .context("Failed to resolve Julia dependency order")?;
    for path in ordered_julia_files {
        if args.verbose {
            println!("  Analyzing: {:?}", path);
        }

        match julia_analyzer.analyze_file(&path) {
            Ok(result) => {
                julia_count += 1;
                combined_result.merge(result);
            }
            Err(e) => {
                eprintln!("Warning: Failed to analyze {:?}: {}", path, e);
            }
        }
    }
    println!("Analyzed {} Julia files", julia_count);

    // Build control flow graph
    println!("\nBuilding control flow graph...");
    let mut cf_analyzer = ControlFlowAnalyzer::new();
    cf_analyzer.build_call_graph(&combined_result);

    // Generate reports
    println!("\nGenerating reports...");
    let report_gen = ReportGenerator::new(output_path.to_string_lossy().to_string());
    report_gen
        .generate_all(
            &combined_result,
            &cf_analyzer,
            &rust_layer_graph,
            &julia_layer_graph,
        )
        .context("Failed to generate reports")?;

    println!("\n✓ Analysis complete!");
    println!("  Total elements: {}", combined_result.elements.len());
    println!("  Rust files: {}", rust_count);
    println!("  Julia files: {}", julia_count);
    println!("  Reports saved to: {:?}", output_path);

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
