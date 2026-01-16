//! Edge Type Classification

use crate::extract::RefType;
use mmsb_core::dag::EdgeType;

/// Map extracted reference type to DAG edge type
pub fn ref_type_to_edge_type(ref_type: RefType) -> EdgeType {
    match ref_type {
        RefType::Import => EdgeType::Data,
        RefType::Module => EdgeType::Control,
        RefType::Call => EdgeType::Data,
    }
}
