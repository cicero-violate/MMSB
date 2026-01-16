//! Query execution and mutation application

use crate::source::SourceBuffer;
use crate::query::QueryPlan;
use crate::mutation::MutationPlan;
use crate::error::EditorError;
use syn::Item;

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
    let matches: Vec<Item> = execute_query(buffer, plan.query())
        .into_iter()
        .cloned()
        .collect();
    
    if matches.is_empty() {
        return Err(EditorError::NoMatches);
    }
    
    // TODO: Apply transformations and update buffer
    Ok(())
}
