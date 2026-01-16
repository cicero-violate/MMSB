pub mod scan;
pub mod diff;
pub mod map;
pub mod error;
pub mod propagation;

use mmsb_core::dag::DependencyGraph;
use mmsb_core::prelude::Delta;
use mmsb_core::dag::StructuralOp;
use structural_code_indexer::fs::SourceFile;

pub use error::CodeEditError;
pub use scan::RepoScan;
pub use diff::RepoDiff;
pub use map::MappedEdit;
pub use propagation::{
    EditIntent, PageIndex, PropagatedDelta,
    extract_intent, index_page, index_snapshot, propagate_edits, rewrite_page,
};

#[derive(Debug, Clone, Default)]
pub struct RepoSnapshot {
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone, Default)]
pub struct CodeEdit {
    pub deltas: Vec<Delta>,
    pub ops: Vec<StructuralOp>,
}

pub fn analyze_code_edit(
    repo_before: &RepoSnapshot,
    repo_after: &RepoSnapshot,
    active_dag: &DependencyGraph,
) -> Result<CodeEdit, CodeEditError> {
    let before_scan = scan::scan_repo(repo_before)?;
    let after_scan = scan::scan_repo(repo_after)?;
    diff::validate_active_dag(active_dag, &before_scan.graph)?;

    let diff = diff::diff_repo(&before_scan, &after_scan);
    let mapped = map::map_edit(&diff)?;

    Ok(CodeEdit {
        deltas: mapped.deltas,
        ops: mapped.ops,
    })
}
