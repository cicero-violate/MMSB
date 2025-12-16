mod types;
mod rust_parser;
mod julia_parser;
mod control_flow;
mod report;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::env;
use walkdir::WalkDir;

use crate::types::AnalysisResult;
use crate::rust_parser::RustAnalyzer;
use crate::julia_parser::JuliaAnalyzer;
use crate::control_flow::ControlFlowAnalyzer;
use crate::report::ReportGenerator;

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
    #[arg(short, long, default_value = "./julia_analyzer.jl")]
    julia_script: PathBuf,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    // Resolve to absolute paths
    let root_path = env::current_dir()?.join(&args.root).canonicalize()?;
    let output_path = env::current_dir()?.join(&args.output).canonicalize()
        .unwrap_or_else(|_| {
            let p = env::current_dir().unwrap().join(&args.output);
            std::fs::create_dir_all(&p).ok();
            p.canonicalize().unwrap_or(p)
        });
    let julia_script_path = env::current_dir()?.join(&args.julia_script).canonicalize()?;
    
    println!("MMSB Intelligence Substrate Analyzer");
    println!("=====================================\n");
    println!("Root directory: {:?}", root_path);
    println!("Output directory: {:?}", output_path);
    println!("Julia script: {:?}\n", julia_script_path);
    
    let rust_analyzer = RustAnalyzer::new(
        root_path.to_string_lossy().to_string()
    );
    
    let julia_analyzer = JuliaAnalyzer::new(
        root_path.to_string_lossy().to_string(),
        julia_script_path.to_string_lossy().to_string(),
    );
    
    let mut combined_result = AnalysisResult::new();
    
    // Scan for Rust files
    println!("Scanning Rust files...");
    let mut rust_count = 0;
    for entry in WalkDir::new(&root_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
        .filter(|e| !e.path().components().any(|c| c.as_os_str() == "target" || c.as_os_str() == "examples"))
    {
        let path = entry.path();
        if args.verbose {
            println!("  Analyzing: {:?}", path);
        }
        
        match rust_analyzer.analyze_file(path) {
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
    println!("\nScanning Julia files...");
    let mut julia_count = 0;
    for entry in WalkDir::new(&root_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "jl"))
        .filter(|e| !e.path().components().any(|c| c.as_os_str() == "test" || c.as_os_str() == "examples"))
    {
        let path = entry.path();
        if args.verbose {
            println!("  Analyzing: {:?}", path);
        }
        
        match julia_analyzer.analyze_file(path) {
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
    report_gen.generate_all(&combined_result, &cf_analyzer)
        .context("Failed to generate reports")?;
    
    println!("\n✓ Analysis complete!");
    println!("  Total elements: {}", combined_result.elements.len());
    println!("  Rust files: {}", rust_count);
    println!("  Julia files: {}", julia_count);
    println!("  Reports saved to: {:?}", output_path);
    
    Ok(())
}
