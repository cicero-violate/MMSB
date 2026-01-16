use crate::error::CodeEditError;
use crate::scan::RepoScan;
use mmsb_core::dag::{DependencyGraph, StructuralOp};
use mmsb_core::types::PageID;
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct FileChange {
    pub path: PathBuf,
    pub content: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct RepoDiff {
    pub file_changes: Vec<FileChange>,
    pub structural_ops: Vec<StructuralOp>,
}

pub fn diff_repo(before: &RepoScan, after: &RepoScan) -> RepoDiff {
    let mut file_changes = Vec::new();
    let mut all_paths = BTreeSet::new();
    all_paths.extend(before.files.keys().cloned());
    all_paths.extend(after.files.keys().cloned());

    for path in all_paths {
        let before_bytes = before.files.get(&path);
        let after_bytes = after.files.get(&path);
        if before_bytes == after_bytes {
            continue;
        }
        let content = after_bytes.map(|bytes| bytes.clone());
        file_changes.push(FileChange { path, content });
    }

    let structural_ops = diff_structural(&before.graph, &after.graph);
    RepoDiff {
        file_changes,
        structural_ops,
    }
}

pub fn validate_active_dag(
    active: &DependencyGraph,
    scanned: &DependencyGraph,
) -> Result<(), CodeEditError> {
    let expected = edge_hash(active);
    let found = edge_hash(scanned);
    if expected != found {
        return Err(CodeEditError::DagMismatch { expected, found });
    }
    Ok(())
}

fn diff_structural(
    before: &DependencyGraph,
    after: &DependencyGraph,
) -> Vec<StructuralOp> {
    let before_edges = edge_set(before);
    let after_edges = edge_set(after);

    let mut ops = Vec::new();

    for (from, to, _) in before_edges.difference(&after_edges) {
        ops.push(StructuralOp::RemoveEdge {
            from: PageID(*from),
            to: PageID(*to),
        });
    }

    for (from, to, edge_type) in after_edges.difference(&before_edges) {
        ops.push(StructuralOp::AddEdge {
            from: PageID(*from),
            to: PageID(*to),
            edge_type: edge_type_from_u8(*edge_type),
        });
    }

    ops
}

fn edge_set(graph: &DependencyGraph) -> BTreeSet<(u64, u64, u8)> {
    graph
        .edges()
        .into_iter()
        .map(|(from, to, edge_type)| (from.0, to.0, edge_type as u8))
        .collect()
}

fn edge_hash(graph: &DependencyGraph) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (from, to, edge_type) in edge_set(graph) {
        hasher.update(from.to_le_bytes());
        hasher.update(to.to_le_bytes());
        hasher.update([edge_type]);
    }
    format!("{:x}", hasher.finalize())
}

fn edge_type_from_u8(value: u8) -> mmsb_core::dag::EdgeType {
    match value {
        0 => mmsb_core::dag::EdgeType::Data,
        1 => mmsb_core::dag::EdgeType::Control,
        2 => mmsb_core::dag::EdgeType::Gpu,
        3 => mmsb_core::dag::EdgeType::Compiler,
        _ => mmsb_core::dag::EdgeType::Data,
    }
}
