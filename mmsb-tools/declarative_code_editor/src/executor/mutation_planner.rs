use crate::buffer::EditBuffer;
use crate::error::EditorError;
use crate::executor::query_executor::execute_query;
use crate::mutation::MutationPlan;
use crate::types::Edit;
use quote::ToTokens;
use syn::Item;

/// Planned edit - matched item + operation + concrete edit
#[derive(Debug, Clone)]
pub struct PlannedEdit {
    pub item_name: String,
    pub edit: Edit,
}

/// Plan mutations from query matches
pub fn plan_mutations(
    plan: &MutationPlan,
    buffer: &EditBuffer,
    allow_empty: bool,
    allow_multiple: bool,
) -> Result<Vec<PlannedEdit>, EditorError> {
    let matches = execute_query(plan.query(), buffer, allow_empty, allow_multiple)?;
    let mut planned = Vec::new();

    for item in matches {
        let item_name = extract_item_name(item);
        
        for op in plan.operations() {
            let new_text = op.apply(item);
            let (start, end) = item_byte_range(item, buffer);
            
            let edit = Edit::new(start, end, new_text);
            planned.push(PlannedEdit { item_name: item_name.clone(), edit });
        }
    }

    // Detect conflicts
    detect_conflicts(&planned)?;

    // Sort by start_byte descending (apply from end to start)
    planned.sort_by(|a, b| b.edit.start_byte.cmp(&a.edit.start_byte));

    Ok(planned)
}

fn extract_item_name(item: &Item) -> String {
    match item {
        Item::Fn(f) => f.sig.ident.to_string(),
        Item::Struct(s) => s.ident.to_string(),
        Item::Enum(e) => e.ident.to_string(),
        Item::Trait(t) => t.ident.to_string(),
        Item::Mod(m) => m.ident.to_string(),
        Item::Const(c) => c.ident.to_string(),
        Item::Static(s) => s.ident.to_string(),
        Item::Type(t) => t.ident.to_string(),
        _ => "unknown".to_string(),
    }
}

fn item_byte_range(item: &Item, buffer: &EditBuffer) -> (u32, u32) {
    // Find item in source by converting to string and searching
    let item_str = item.to_token_stream().to_string();
    let source = buffer.source();
    
    // Simple heuristic: find the item text in source
    // In production, would use span information from syn
    if let Some(pos) = source.find(&item_str) {
        let start = pos as u32;
        let end = start + item_str.len() as u32;
        (start, end)
    } else {
        // Fallback: entire file (this is a limitation of the simple approach)
        (0, source.len() as u32)
    }
}

fn detect_conflicts(planned: &[PlannedEdit]) -> Result<(), EditorError> {
    let mut ranges = Vec::new();
    
    for edit in planned {
        let (start, end) = edit.edit.byte_range();
        
        for &(seen_start, seen_end) in &ranges {
            if start < seen_end && seen_start < end {
                return Err(EditorError::ConflictingEdits(start, end, seen_start, seen_end));
            }
        }
        
        ranges.push((start, end));
    }
    
    Ok(())
}
