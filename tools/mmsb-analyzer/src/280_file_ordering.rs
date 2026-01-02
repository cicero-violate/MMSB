//! File-level dependency ordering using a DAG.

use anyhow::Result;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use crate::dependency::build_module_map;
use crate::cluster_010::extract_dependencies;

pub use crate::cluster_001::{ordered_by_name, topological_sort};
pub use crate::cluster_010::build_dependency_map;

#[derive(Clone, Debug)]
pub struct DirectoryMove {
    pub from: PathBuf,
    pub to: PathBuf,
}

#[allow(dead_code)]
pub struct DagCache {
    graph: DiGraph<PathBuf, ()>,
    node_map: HashMap<PathBuf, NodeIndex>,
    topo_order: Vec<NodeIndex>,
    last_modified: HashMap<PathBuf, SystemTime>,
    file_set: HashSet<PathBuf>,
    module_map: HashMap<String, PathBuf>,
}

#[allow(dead_code)]
impl DagCache {
    pub fn new(files: &[PathBuf]) -> Result<Self> {
        let file_set: HashSet<PathBuf> = files.iter().cloned().collect();
        let module_map = build_module_map(files);
        let dep_map = build_dependency_map(files, &file_set, &module_map)?;
        let (graph, node_map) = build_file_dag(files, &dep_map);
        let topo_order = topological_sort(&graph).unwrap_or_else(|_| {
            graph
                .node_indices()
                .collect::<Vec<_>>()
        });

        let mut last_modified = HashMap::new();
        for file in files {
            if let Ok(meta) = fs::metadata(file) {
                if let Ok(modified) = meta.modified() {
                    last_modified.insert(file.clone(), modified);
                }
            }
        }

        Ok(Self {
            graph,
            node_map,
            topo_order,
            last_modified,
            file_set,
            module_map,
        })
    }

    pub fn incremental_update(&mut self, changed_files: &[PathBuf]) -> Result<()> {
        for file in changed_files {
            if !self.file_set.contains(file) {
                continue;
            }
            let Some(&node) = self.node_map.get(file) else {
                continue;
            };
            let old_edges: Vec<_> = self.graph.edges(node).map(|e| e.id()).collect();
            for edge in old_edges {
                self.graph.remove_edge(edge);
            }

            let deps = extract_dependencies(file, &self.file_set, &self.module_map)?;
            for dep in deps {
                if let Some(&dep_node) = self.node_map.get(&dep) {
                    self.graph.add_edge(dep_node, node, ());
                }
            }

            if let Ok(meta) = fs::metadata(file) {
                if let Ok(modified) = meta.modified() {
                    self.last_modified.insert(file.clone(), modified);
                }
            }
        }

        if let Ok(order) = topological_sort(&self.graph) {
            self.topo_order = order;
        }
        Ok(())
    }

    pub fn topo_files(&self) -> Vec<PathBuf> {
        self.topo_order
            .iter()
            .map(|idx| self.graph[*idx].clone())
            .collect()
    }
}

#[allow(dead_code)]
pub fn parallel_build_file_dag(directories: &[PathBuf]) -> Result<DiGraph<PathBuf, ()>> {
    let subgraphs: Vec<DiGraph<PathBuf, ()>> = directories
        .par_iter()
        .map(|dir| crate::dependency::build_directory_dag(dir))
        .collect::<Result<_>>()?;

    let mut merged = DiGraph::new();
    let mut node_map: HashMap<PathBuf, NodeIndex> = HashMap::new();

    for subgraph in subgraphs {
        for node in subgraph.node_indices() {
            let file = subgraph[node].clone();
            node_map.entry(file.clone()).or_insert_with(|| merged.add_node(file));
        }
        for edge in subgraph.edge_indices() {
            if let Some((src, dst)) = subgraph.edge_endpoints(edge) {
                let src_file = subgraph[src].clone();
                let dst_file = subgraph[dst].clone();
                let src_node = *node_map.get(&src_file).expect("missing source node");
                let dst_node = *node_map.get(&dst_file).expect("missing target node");
                merged.add_edge(src_node, dst_node, ());
            }
        }
    }

    Ok(merged)
}

// Re-exported from other modules for file ordering utilities
pub(crate) use crate::cluster_001::build_entries;
pub(crate) use crate::cluster_011::build_file_dag;
pub(crate) use crate::cluster_001::detect_cycles;
