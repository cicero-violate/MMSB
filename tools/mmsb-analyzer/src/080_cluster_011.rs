//! Cluster 011: File ordering and analysis utilities extracted from core modules.
//!
//! This module consolidates:
//! - Module name normalization
//! - Module path resolution
//! - Directory/file graph utilities

use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub fn build_module_map(files: &[PathBuf]) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    for file in files {
        if let Some(stem) = file.file_stem().and_then(|s| s.to_str()) {
            let normalized = crate::cluster_010::normalize_module_name(stem);
            map.insert(normalized.clone(), file.clone());
            if stem == "mod" {
                if let Some(parent) = file.parent().and_then(|p| p.file_name()) {
                    if let Some(name) = parent.to_str() {
                        map.insert(crate::cluster_010::normalize_module_name(name), file.clone());
                    }
                }
            }
        }
    }
    map
}

pub fn resolve_path(
    candidate: &Path,
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    if file_set.contains(candidate) {
        return Some(candidate.to_path_buf());
    }
    if let Some(file_name) = candidate.file_stem().and_then(|s| s.to_str()) {
        let key = crate::cluster_010::normalize_module_name(file_name);
        if let Some(path) = module_map.get(&key) {
            return Some(path.clone());
        }
    }
    None
}

pub fn build_directory_dag(dir: &PathBuf) -> Result<DiGraph<PathBuf, ()>> {
    let files: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "rs" || ext == "jl")
                .unwrap_or(false)
        })
        .map(|entry| entry.into_path())
        .collect();

    let file_set: HashSet<PathBuf> = files.iter().cloned().collect();
    let module_map = build_module_map(&files);
    let dep_map = crate::cluster_010::build_dependency_map(&files, &file_set, &module_map)?;
    let (graph, _) = build_file_dag(&files, &dep_map);
    Ok(graph)
}

pub fn build_file_dependency_graph(files: &[PathBuf]) -> Result<DiGraph<PathBuf, ()>> {
    let file_set: HashSet<PathBuf> = files.iter().cloned().collect();
    let module_map = build_module_map(files);
    let dep_map = crate::cluster_010::build_dependency_map(files, &file_set, &module_map)?;
    let (graph, _) = build_file_dag(files, &dep_map);
    Ok(graph)
}

pub fn export_program_cfg_to_path(
    result: &crate::types::AnalysisResult,
    call_edges: &[(String, String)],
    output_path: &Path,
) -> std::io::Result<()> {
    use crate::types::ProgramCFG;

    let mut program_cfg = ProgramCFG {
        functions: HashMap::new(),
        call_edges: Vec::new(),
    };

    for cfg in &result.cfgs {
        program_cfg
            .functions
            .insert(cfg.function.clone(), cfg.clone());
    }

    for (caller, callee) in call_edges {
        let caller_name = caller.split("::").last().unwrap_or(caller).to_string();
        let callee_name = callee.split("::").last().unwrap_or(callee).to_string();
        if program_cfg.functions.contains_key(&caller_name)
            && program_cfg.functions.contains_key(&callee_name)
        {
            program_cfg.call_edges.push((caller_name, callee_name));
        }
    }

    let cfg_dir = output_path.join("30_cfg");
    std::fs::create_dir_all(&cfg_dir)?;
    let dot_path = cfg_dir.join("complete_program.dot");
    crate::cluster_001::export_complete_program_dot(
        &program_cfg,
        dot_path.to_string_lossy().as_ref(),
    )?;

    #[cfg(feature = "png")]
    {
        let png_path = cfg_dir.join("complete_program.png");
        if let (Some(dot_path_str), Some(png_path_str)) =
            (dot_path.to_str(), png_path.to_str())
        {
            let _ = std::process::Command::new("dot")
                .args(["-Tpng", dot_path_str, "-o", png_path_str])
                .status();
        }
    }

    Ok(())
}

pub(crate) fn build_file_dag(
    files: &[PathBuf],
    dep_map: &HashMap<PathBuf, Vec<PathBuf>>,
) -> (DiGraph<PathBuf, ()>, HashMap<PathBuf, NodeIndex>) {
    let mut graph = DiGraph::new();
    let mut node_map = HashMap::new();

    for file in files {
        let node = graph.add_node(file.clone());
        node_map.insert(file.clone(), node);
    }

    for (file, deps) in dep_map {
        if let Some(&file_node) = node_map.get(file) {
            for dep in deps {
                if let Some(&dep_node) = node_map.get(dep) {
                    graph.add_edge(dep_node, file_node, ());
                }
            }
        }
    }

    (graph, node_map)
}
