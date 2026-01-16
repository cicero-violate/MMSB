use crate::intent::EditIntent;
use crate::error::EditorError;
use mmsb_core::dag::DependencyGraph;
use mmsb_core::prelude::Delta;
use mmsb_core::types::PageID;
use mmsb_judgment::JudgmentToken;
use std::collections::HashMap;

/// Bridge to structural_code_editor propagation engine
pub struct PropagationBridge;

impl PropagationBridge {
    /// Trigger propagation using structural_code_editor
    ///
    /// This requires:
    /// 1. JudgmentToken (authority to emit derived deltas)
    /// 2. Active DAG snapshot (read-only)
    /// 3. Page indices (for matching intents)
    /// 4. Source store (for rewriting)
    pub fn propagate(
        root_page: PageID,
        intents: &[EditIntent],
        graph: &DependencyGraph,
        judgment: &JudgmentToken,
    ) -> Result<Vec<Delta>, EditorError> {
        // Convert EditIntent to structural_code_editor::EditIntent
        let structural_intents = Self::convert_intents(intents);
        
        // Build index store and source store
        // TODO: In production, these would come from the actual page store
        let index_store = HashMap::new();
        let source_store = HashMap::new();
        
        // Call structural_code_editor propagation
        let propagated = structural_code_editor::propagation::propagate_edits(
            root_page,
            &structural_intents,
            graph,
            &index_store,
            &source_store,
            judgment,
        );
        
        // Extract deltas from propagated results
        let deltas = propagated.into_iter().map(|p| p.delta).collect();
        
        Ok(deltas)
    }
    
    /// Convert declarative EditIntent to structural EditIntent
    fn convert_intents(intents: &[EditIntent]) -> Vec<structural_code_editor::EditIntent> {
        intents
            .iter()
            .filter_map(|intent| match intent {
                EditIntent::RenameSymbol { old, new } => {
                    Some(structural_code_editor::EditIntent::RenameSymbol {
                        old: old.clone(),
                        new: new.clone(),
                    })
                }
                EditIntent::DeleteSymbol { name } => {
                    Some(structural_code_editor::EditIntent::DeleteSymbol {
                        name: name.clone(),
                    })
                }
                EditIntent::SignatureChange { name } => {
                    Some(structural_code_editor::EditIntent::SignatureChange {
                        name: name.clone(),
                    })
                }
                EditIntent::AddSymbol { name, kind: _ } => {
                    Some(structural_code_editor::EditIntent::AddSymbol {
                        name: name.clone(),
                    })
                }
                // ImportChange and ModuleChange don't propagate
                _ => None,
            })
            .collect()
    }
}
