use crate::error::CodeEditError;
use crate::RepoSnapshot;
use mmsb_core::dag::DependencyGraph;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use structural_code_indexer::extract::extract_rust_refs;
use structural_code_indexer::graph::build_dependency_graph;
use structural_code_indexer::index::{PageIndex, SymbolIndex};

#[derive(Debug, Clone)]
pub struct RepoScan {
    pub files: BTreeMap<PathBuf, Vec<u8>>,
    pub graph: DependencyGraph,
}

pub fn scan_repo(snapshot: &RepoSnapshot) -> Result<RepoScan, CodeEditError> {
    let mut files = BTreeMap::new();
    let mut refs = Vec::new();
    let mut page_index = PageIndex::new();
    let mut symbol_index = SymbolIndex::new();

    for file in &snapshot.files {
        let path = canonicalize_path(&file.path);
        if files.contains_key(&path) {
            return Err(CodeEditError::DuplicatePath(path.to_string_lossy().to_string()));
        }
        files.insert(path.clone(), file.content.as_bytes().to_vec());
        refs.extend(extract_rust_refs(path, &file.content));
    }

    let graph = build_dependency_graph(&refs, &mut page_index, &mut symbol_index);
    Ok(RepoScan { files, graph })
}

fn canonicalize_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir | std::path::Component::ParentDir))
        .collect()
}
