use crate::dag::dependency_graph::DependencyGraph;
use crate::structural::structural_types::StructuralOp;
use crate::types::EdgeType;

pub fn append_structural_record(_ops: &[StructuralOp]) -> std::io::Result<()> {
    Ok(())
}
pub fn default_structural_log_path() -> String {
    "/tmp/structural.log".to_string()
pub fn replay_structural_log(_path: &str) -> std::io::Result<DependencyGraph> {
    Ok(DependencyGraph::new())
