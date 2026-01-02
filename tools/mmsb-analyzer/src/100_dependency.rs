//! Dependency-aware ordering and layer graph extraction.

pub use crate::cluster_010::order_julia_files_by_dependency;
#[allow(unused_imports)]
pub use crate::cluster_001::{detect_layer, julia_entry_paths, order_rust_files_by_dependency};
pub use crate::cluster_011::build_module_map;
pub use crate::cluster_001::{build_directory_entry_map, collect_naming_warnings};
#[allow(unused_imports)]
pub use crate::cluster_010::{extract_julia_dependencies, extract_rust_dependencies};
pub use crate::cluster_001::{analyze_file_ordering, naming_score_for_file};
pub use crate::cluster_011::{build_directory_dag, build_file_dependency_graph};
use std::collections::BTreeSet;
use std::path::PathBuf;
use syn::UseTree;

#[allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootState {
    Start,
    AfterRoot,
}

#[derive(Debug, Clone)]
pub struct LayerGraph {
    pub ordered_layers: Vec<String>,
    pub edges: Vec<LayerEdge>,
    pub cycles: Vec<String>,
    pub unresolved: Vec<UnresolvedDependency>,
}

impl Default for LayerGraph {
    fn default() -> Self {
        LayerGraph {
            ordered_layers: Vec::new(),
            edges: Vec::new(),
            cycles: Vec::new(),
            unresolved: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerEdge {
    pub from: String,
    pub to: String,
    pub references: Vec<ReferenceDetail>,
    pub violation: bool,
}

#[derive(Debug, Clone)]
pub struct UnresolvedDependency {
    pub file: PathBuf,
    pub reference: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct ReferenceDetail {
    pub file: PathBuf,
    pub reference: String,
}

pub fn collect_roots(tree: &UseTree, state: RootState, acc: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => {
            let ident = path.ident.to_string();
            if state == RootState::Start && matches!(ident.as_str(), "crate" | "self" | "super") {
                collect_roots(&path.tree, RootState::AfterRoot, acc);
            } else if state == RootState::AfterRoot {
                acc.insert(ident);
            } else {
                acc.insert(ident);
            }
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                collect_roots(tree, state, acc);
            }
        }
        UseTree::Name(name) => {
            acc.insert(name.ident.to_string());
        }
        UseTree::Rename(rename) => {
            acc.insert(rename.ident.to_string());
        }
        UseTree::Glob(_) => {}
    }
}
