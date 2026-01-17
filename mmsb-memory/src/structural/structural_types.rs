use crate::types::PageID;
use crate::dag::EdgeType;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum StructuralOp {
    AddEdge {
        from: PageID,
        to: PageID,
        edge_type: EdgeType,
    },
    RemoveEdge {
        from: PageID,
        to: PageID,
    },
}

impl StructuralOp {
    pub fn from_page(&self) -> PageID {
        match self {
            StructuralOp::AddEdge { from, .. } => *from,
            StructuralOp::RemoveEdge { from, .. } => *from,
        }
    }

    pub fn to_page(&self) -> PageID {
        match self {
            StructuralOp::AddEdge { to, .. } => *to,
            StructuralOp::RemoveEdge { to, .. } => *to,
        }
    }

    pub fn is_add(&self) -> bool {
        matches!(self, StructuralOp::AddEdge { .. })
    }

    pub fn is_remove(&self) -> bool {
        matches!(self, StructuralOp::RemoveEdge { .. })
    }
}
use crate::types::EdgeType;
