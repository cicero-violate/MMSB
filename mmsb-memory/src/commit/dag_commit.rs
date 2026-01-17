use crate::dag::DependencyGraph;
use crate::structural::{StructuralOp, has_cycle, GraphValidator};
use crate::outcome::DagCommitError;
use crate::structural::shadow_graph::ShadowPageGraph;
use crate::replay::dag_log::{append_structural_record, default_structural_log_path};
use sha2::{Digest, Sha256};
pub fn build_dependency_graph(ops: &[StructuralOp]) -> DependencyGraph {
let mut dag = DependencyGraph::new();
dag.apply_ops(ops);
dag
}
pub(crate) fn apply_ops_unchecked(dag: &mut DependencyGraph, ops: &[StructuralOp]) {
dag.apply_ops(ops);
}
pub fn commit_structural_delta(
dag: &mut DependencyGraph,
ops: &[StructuralOp],
) -> Result<(), DagCommitError> {
if ops.is_empty() {
return Err(DagCommitError::EmptyOperations);
}
for op in ops {
if !dag.contains_page(op.from_page()) {
return Err(DagCommitError::InvalidPageReference(op.from_page()));
}
if !dag.contains_page(op.to_page()) {
return Err(DagCommitError::InvalidPageReference(op.to_page()));
}
}
let shadow = ShadowPageGraph::from_dag_and_ops(dag, ops);
if has_cycle(&shadow) {
return Err(DagCommitError::CycleDetected);
}
let validator = GraphValidator::new(&shadow);
let report = validator.detect_cycles();
if report.has_cycle {
return Err(DagCommitError::CycleDetected);
}
apply_ops_unchecked(dag, ops);
let path = default_structural_log_path();
append_structural_record(&path, ops)
}
fn compute_ops_hash_local(ops: &[StructuralOp]) -> String {
let mut hasher = Sha256::new();
for op in ops {
hasher.update(format!("{:?}", op).as_bytes());  // Use Debug for hash
}
format!("{:x}", hasher.finalize())
}
#[cfg(test)]
mod tests {
use super::{commit_structural_delta, compute_ops_hash_local};
use crate::structural::{EdgeType, StructuralOp};
use crate::outcome::DagCommitError;
use crate::page::PageID;
#[test]
fn commit_rejects_empty_ops() {
let mut dag = DependencyGraph::new();
let ops = vec![];
let err = commit_structural_delta(&mut dag, &ops)
.expect_err("expected empty ops error");
assert!(matches!(err, DagCommitError::EmptyOperations));
}
#[test]
fn commit_rejects_invalid_page_reference() {
let mut dag = DependencyGraph::new();
let ops = vec![StructuralOp::AddEdge {
from: PageID(1),
to: PageID(2),
edge_type: EdgeType::Data,
}];
let err = commit_structural_delta(&mut dag, &ops)
.expect_err("expected invalid page reference");
assert!(matches!(err, DagCommitError::InvalidPageReference(_) ));
}
}
