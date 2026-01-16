//! Bridge Orchestrator
//!
//! Orchestrates the full flow from declarative edits to MMSB authority pipelines.
//!
//! Flow:
//! 1. Execute declarative mutation (query + mutate + upsert)
//! 2. Extract semantic intent from planned edits
//! 3. Classify into structural vs state changes
//! 4. Route to appropriate pipelines

use crate::error::EditorError;
use crate::mutation::MutationPlan;
use crate::executor::apply_mutation;
use crate::bridge::{StructuralClassifier, BridgedOutput};
use mmsb_core::types::PageID;
use std::path::PathBuf;

/// Orchestrates declarative edit → MMSB pipelines
pub struct BridgeOrchestrator;

impl BridgeOrchestrator {
    /// Apply mutation to source and produce bridged output
    ///
    /// This is the main entry point that:
    /// 1. Apply mutation to source text
    /// 2. Extract semantic intent from before/after
    /// 3. Classify into structural/state
    /// 4. Build Delta and StructuralOps
    /// 5. Return output ready for MMSB commits
    pub fn execute_and_bridge(
        source_before: &str,
        plan: &MutationPlan,
        page_id: PageID,
        file_path: &PathBuf,
    ) -> Result<BridgedOutput, EditorError> {
        // Step 1: Apply mutation to get new source
        let source_after = apply_mutation(source_before, plan)?;
        
        // Step 2: Extract semantic intent (simplified for now)
        let intents = Vec::new(); // TODO: implement intent extraction from source diff
        
        // Step 3: Classify and build Delta + StructuralOps
        let (page_deltas, structural_ops) = StructuralClassifier::classify(
            &intents,
            page_id,
            file_path,
            &source_after,
        )?;
        
        // Step 4: Create bridged output
        Ok(BridgedOutput::new(intents, page_deltas, structural_ops))
    }
    
}
