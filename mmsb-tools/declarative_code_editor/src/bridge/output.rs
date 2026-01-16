use mmsb_core::prelude::Delta;
use mmsb_core::dag::StructuralOp;
use crate::intent::EditIntent;

/// Pipeline routing decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineRoute {
    /// STATE PIPELINE: commit_delta → TLog → propagation
    State,
    
    /// STRUCTURAL PIPELINE: ShadowGraph → validate → commit_structural_delta
    Structural,
    
    /// BOTH: structural first, then state
    Both,
}

/// Bridged output - ready for MMSB authority pipelines
#[derive(Debug, Clone)]
pub struct BridgedOutput {
    /// Extracted semantic intent
    pub intents: Vec<EditIntent>,
    
    /// Page deltas (STATE PIPELINE)
    pub page_deltas: Vec<Delta>,
    
    /// Structural ops (STRUCTURAL PIPELINE)
    pub structural_ops: Vec<StructuralOp>,
    
    /// Pipeline routing decision
    pub route: PipelineRoute,
}

impl BridgedOutput {
    pub fn new(
        intents: Vec<EditIntent>,
        page_deltas: Vec<Delta>,
        structural_ops: Vec<StructuralOp>,
    ) -> Self {
        let route = determine_route(&intents);
        Self {
            intents,
            page_deltas,
            structural_ops,
            route,
        }
    }
    
    /// Check if requires structural commit
    pub fn needs_structural_commit(&self) -> bool {
        matches!(self.route, PipelineRoute::Structural | PipelineRoute::Both)
    }
    
    /// Check if requires state commit
    pub fn needs_state_commit(&self) -> bool {
        matches!(self.route, PipelineRoute::State | PipelineRoute::Both)
    }
}

fn determine_route(intents: &[EditIntent]) -> PipelineRoute {
    use crate::intent::IntentCategory;
    
    let has_state = intents.iter().any(|i| {
        matches!(i.category(), IntentCategory::State | IntentCategory::Both)
    });
    let has_structural = intents.iter().any(|i| {
        matches!(i.category(), IntentCategory::Structural | IntentCategory::Both)
    });
    
    match (has_state, has_structural) {
        (true, true) => PipelineRoute::Both,
        (true, false) => PipelineRoute::State,
        (false, true) => PipelineRoute::Structural,
        (false, false) => PipelineRoute::State, // default to state
    }
}
