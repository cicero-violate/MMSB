//! Query execution and mutation application

use crate::source::SourceBuffer;
use crate::query::QueryPlan;
use crate::mutation::MutationPlan;
use crate::error::EditorError;
use syn::Item;

// TODO: Executor improvements
// - Conflict detection (overlapping mutations)
// - Ordering strategy (deterministic application order)
// - Rollback support (transaction semantics)
// - Dry-run mode (preview without applying)
// - Diff generation (show before/after with context)
// - Incremental updates (reparse only changed parts)
// - Formatting preservation (keep original formatting where possible)

/// Execute query against source buffer
pub fn execute_query<'a>(buffer: &'a SourceBuffer, plan: &QueryPlan) -> Vec<&'a Item> {
    let file = buffer.ast();
    
    let mut results = Vec::new();
    for item in &file.items {
        if plan.matches(item) {
            results.push(item);
        }
    }
    
    results
}

/// Apply mutation to buffer (modifies in-place)
pub fn apply_mutation(buffer: &mut SourceBuffer, plan: &MutationPlan) -> Result<(), EditorError> {
    // Find matches
    let match_indices: Vec<usize> = {
        let file = buffer.ast();
        let mut indices = Vec::new();
        for (idx, item) in file.items.iter().enumerate() {
            if plan.query().matches(item) {
                indices.push(idx);
            }
        }
        indices
    };
    
    if match_indices.is_empty() {
        return Err(EditorError::NoMatches);
    }
    
    // Clone the AST file and apply transformations
    let mut new_file = buffer.ast().clone();
    
    // Apply operations to matched items
    for &idx in &match_indices {
        let item = &new_file.items[idx];
        
        // Apply all operations to get replacement text
        let mut replacement = String::new();
        for op in plan.operations() {
            replacement = op.apply(item);
        }
        
        // Parse replacement as new item
        let new_item = syn::parse_str::<Item>(&replacement)
            .map_err(|e| EditorError::ParseError(format!("Invalid replacement: {}", e)))?;
        
        new_file.items[idx] = new_item;
    }
    
    // Serialize the modified AST back to source
    let new_source = quote::quote!(#new_file).to_string();
    
    // Update buffer with new source (this will re-parse)
    buffer.update(new_source)?;
    
    Ok(())
}
