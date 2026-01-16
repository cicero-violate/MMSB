//! Bridge Orchestrator
//!
//! Orchestrates the full flow from declarative edits to MMSB authority pipelines.
//!
//! Flow:
//! 1. Execute declarative mutation (query + mutate + upsert)
//! 2. Extract semantic intent from planned edits
//! 3. Classify into structural vs state changes
//! 4. Route to appropriate pipelines
//! 5. Trigger propagation (if state changes + judgment provided)

use crate::buffer::EditBuffer;
use crate::error::EditorError;
use crate::mutation::MutationPlan;
use crate::executor::mutation_planner::plan_mutations;
use crate::bridge::{IntentBridge, StructuralClassifier, PropagationBridge, BridgedOutput};
use mmsb_core::dag::DependencyGraph;
use mmsb_core::types::PageID;
use mmsb_judgment::JudgmentToken;
use std::path::PathBuf;

/// Orchestrates declarative edit → MMSB pipelines
pub struct BridgeOrchestrator;

impl BridgeOrchestrator {
    /// Execute declarative mutation and produce bridged output
    ///
    /// This is the main entry point that:
    /// 1. Plans mutations from query
    /// 2. Extracts semantic intent
    /// 3. Classifies into structural/state
    /// 4. Returns output ready for MMSB commits
    pub fn execute_and_bridge(
        plan: &MutationPlan,
        buffer: &mut EditBuffer,
        page_id: PageID,
        file_path: &PathBuf,
        allow_empty: bool,
        allow_multiple: bool,
    ) -> Result<BridgedOutput, EditorError> {
        // Step 1: Plan mutations
        let planned = plan_mutations(plan, buffer, allow_empty, allow_multiple)?;
        
        // Step 2: Extract semantic intent
        let intents = IntentBridge::extract_from_planned(&planned, buffer)?;
        
        // Step 3: Apply edits to buffer
        for plan_edit in &planned {
            buffer.add_edit(plan_edit.edit.clone());
        }
        
        let source_after = buffer.render();
        let edits: Vec<_> = planned.iter().map(|p| p.edit.clone()).collect();
        
        // Step 4: Classify into structural vs state
        let (page_deltas, structural_ops) = StructuralClassifier::classify(
            &intents,
            &edits,
            page_id,
            file_path,
            &source_after,
        )?;
        
        // Step 5: Create bridged output
        Ok(BridgedOutput::new(intents, page_deltas, structural_ops))
    }
    
    /// Execute with propagation
    ///
    /// Requires:
    /// - JudgmentToken for state propagation
    /// - DependencyGraph snapshot (read-only)
    ///
    /// Returns both direct changes AND propagated deltas
    pub fn execute_with_propagation(
        plan: &MutationPlan,
        buffer: &mut EditBuffer,
        page_id: PageID,
        file_path: &PathBuf,
        graph: &DependencyGraph,
        judgment: &JudgmentToken,
        allow_empty: bool,
        allow_multiple: bool,
    ) -> Result<BridgedOutputWithPropagation, EditorError> {
        // Execute and bridge
        let output = Self::execute_and_bridge(
            plan,
            buffer,
            page_id,
            file_path,
            allow_empty,
            allow_multiple,
        )?;
        
        // Propagate if state changes
        let propagated_deltas = if output.needs_state_commit() {
            PropagationBridge::propagate(page_id, &output.intents, graph, judgment)?
        } else {
            Vec::new()
        };
        
        Ok(BridgedOutputWithPropagation {
            direct: output,
            propagated_deltas,
        })
    }
}

/// Output with propagation results
#[derive(Debug, Clone)]
pub struct BridgedOutputWithPropagation {
    /// Direct changes from declarative mutation
    pub direct: BridgedOutput,
    
    /// Propagated deltas from DAG traversal
    pub propagated_deltas: Vec<mmsb_core::prelude::Delta>,
}

impl BridgedOutputWithPropagation {
    /// Get all deltas (direct + propagated)
    pub fn all_deltas(&self) -> Vec<mmsb_core::prelude::Delta> {
        let mut all = self.direct.page_deltas.clone();
        all.extend(self.propagated_deltas.clone());
        all
    }
}
