//! Query execution and mutation application

use crate::query::QueryPlan;
use crate::mutation::MutationPlan;
use crate::error::EditorError;

/// Execute query against source text
pub fn execute_query(source: &str, plan: &QueryPlan) -> Result<Vec<syn::Item>, EditorError> {
    let file = syn::parse_file(source)
        .map_err(|e| EditorError::ParseError(e.to_string()))?;
    
    let mut results = Vec::new();
    for item in &file.items {
        if plan.matches(item) {
            results.push(item.clone());
        }
    }
    
    Ok(results)
}

/// Apply mutation to source text, producing new source
pub fn apply_mutation(source: &str, plan: &MutationPlan) -> Result<String, EditorError> {
    let _file = syn::parse_file(source)
        .map_err(|e| EditorError::ParseError(e.to_string()))?;
    
    // Find matches
    let matches = execute_query(source, plan.query())?;
    
    if matches.is_empty() {
        return Err(EditorError::NoMatches);
    }
    
    // Apply operations to matched items
    let mut new_source = source.to_string();
    
    for item in matches {
        for op in plan.operations() {
            let transformed = op.apply(&item);
            // Replace in source
            // Simplified: would use proper span-based replacement
            let item_str = quote::quote!(#item).to_string();
            new_source = new_source.replace(&item_str, &transformed);
        }
    }
    
    Ok(new_source)
}
