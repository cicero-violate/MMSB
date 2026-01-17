use crate::types::PageID;
use crate::types::EdgeType;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StructuralOp {
    AddEdge {
        from: PageID,
        to: PageID,
        edge_type: EdgeType,
    },
    RemoveEdge {
}
impl StructuralOp {
    pub fn from_page(&self) -> PageID {
        match self {
            StructuralOp::AddEdge { from, .. } => *from,
            StructuralOp::RemoveEdge { from, .. } => *from,
        }
    }
    pub fn to_page(&self) -> PageID {
            StructuralOp::AddEdge { to, .. } => *to,
            StructuralOp::RemoveEdge { to, .. } => *to,
    pub fn is_add(&self) -> bool {
        matches!(self, StructuralOp::AddEdge { .. })
    pub fn is_remove(&self) -> bool {
        matches!(self, StructuralOp::RemoveEdge { .. })
