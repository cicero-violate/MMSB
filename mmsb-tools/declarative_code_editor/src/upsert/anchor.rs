use crate::query::{ItemKind, QueryPlan};

/// Anchor position for insert-on-miss
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorPosition {
    Before,
    After,
}

/// Anchor target specification
#[derive(Debug, Clone)]
pub enum AnchorTarget {
    /// Top of file
    Top,

    /// Bottom of file
    Bottom,

    /// Specific item by kind and name
    Item { kind: ItemKind, name: String },

    /// Query-based target
    Query(QueryPlan),
}

/// Full anchor specification
#[derive(Debug, Clone)]
pub struct AnchorSpec {
    pub position: AnchorPosition,
    pub target: AnchorTarget,
}

impl AnchorSpec {
    pub fn before(target: AnchorTarget) -> Self {
        Self {
            position: AnchorPosition::Before,
            target,
        }
    }

    pub fn after(target: AnchorTarget) -> Self {
        Self {
            position: AnchorPosition::After,
            target,
        }
    }

    pub fn before_item(kind: ItemKind, name: impl Into<String>) -> Self {
        Self {
            position: AnchorPosition::Before,
            target: AnchorTarget::Item {
                kind,
                name: name.into(),
            },
        }
    }

    pub fn after_item(kind: ItemKind, name: impl Into<String>) -> Self {
        Self {
            position: AnchorPosition::After,
            target: AnchorTarget::Item {
                kind,
                name: name.into(),
            },
        }
    }
}
