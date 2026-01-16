//! Mutation safety features
//!
//! Provides conflict detection and dry-run preview mode to prevent
//! accidental data loss and allow inspection before committing changes.

use crate::source::SourceBuffer;
use crate::mutation::MutationPlan;
use crate::error::EditorError;
use syn::Item;
use std::collections::HashSet;

/// Preview result showing what would change without applying
#[derive(Debug, Clone)]
pub struct MutationPreview {
    pub matched_items: Vec<ItemSummary>,
    pub would_modify: usize,
    pub source_before: String,
    pub source_after: String,
}

#[derive(Debug, Clone)]
pub struct ItemSummary {
    pub kind: String,
    pub name: String,
    pub line_start: usize,
}

impl MutationPreview {
    pub fn show_diff(&self) -> String {
        format!(
            "Preview:\n  Matched: {} items\n  Would modify: {} items\n\nBefore:\n{}\n\nAfter:\n{}",
            self.matched_items.len(),
            self.would_modify,
            &self.source_before,
            &self.source_after
        )
    }
}

/// Preview mutation without applying it
pub fn preview_mutation(
    buffer: &SourceBuffer,
    plan: &MutationPlan,
) -> Result<MutationPreview, EditorError> {
    // Find matches
    let file = buffer.ast();
    let mut matched_items = Vec::new();
    let mut match_indices = Vec::new();
    
    for (idx, item) in file.items.iter().enumerate() {
        if plan.query().matches(item) {
            match_indices.push(idx);
            matched_items.push(item_to_summary(item, idx));
        }
    }
    
    if match_indices.is_empty() {
        return Err(EditorError::NoMatches);
    }
    
    // Create temporary buffer to simulate changes
    let mut temp_buffer = buffer.clone();
    
    // Apply mutation to temp buffer
    let mut new_file = temp_buffer.ast().clone();
    
    for &idx in &match_indices {
        let item = &new_file.items[idx];
        let mut replacement = String::new();
        for op in plan.operations() {
            replacement = op.apply(item);
        }
        
        let new_item = syn::parse_str::<Item>(&replacement)
            .map_err(|e| EditorError::ParseError(format!("Invalid replacement: {}", e)))?;
        
        new_file.items[idx] = new_item;
    }
    
    let new_source = quote::quote!(#new_file).to_string();
    
    Ok(MutationPreview {
        matched_items,
        would_modify: match_indices.len(),
        source_before: buffer.source().to_string(),
        source_after: new_source,
    })
}

fn item_to_summary(item: &Item, line: usize) -> ItemSummary {
    let (kind, name) = match item {
        Item::Fn(f) => ("function", f.sig.ident.to_string()),
        Item::Struct(s) => ("struct", s.ident.to_string()),
        Item::Enum(e) => ("enum", e.ident.to_string()),
        Item::Trait(t) => ("trait", t.ident.to_string()),
        Item::Impl(i) => {
            let name = i.self_ty.clone();
            ("impl", quote::quote!(#name).to_string())
        }
        Item::Mod(m) => {
            let name = m.ident.to_string();
            ("mod", name)
        }
        Item::Const(c) => ("const", c.ident.to_string()),
        Item::Static(s) => ("static", s.ident.to_string()),
        Item::Type(t) => ("type", t.ident.to_string()),
        Item::Use(_) => ("use", "...".to_string()),
        _ => ("unknown", "?".to_string()),
    };
    
    ItemSummary {
        kind: kind.to_string(),
        name,
        line_start: line,
    }
}

/// Detect if multiple mutations would conflict
#[derive(Debug, Clone)]
pub struct ConflictDetector {
    affected_items: HashSet<usize>,
}

impl ConflictDetector {
    pub fn new() -> Self {
        Self {
            affected_items: HashSet::new(),
        }
    }
    
    /// Check if plan would conflict with previously added plans
    pub fn check_conflict(
        &mut self,
        buffer: &SourceBuffer,
        plan: &MutationPlan,
    ) -> Result<(), EditorError> {
        let file = buffer.ast();
        
        for (idx, item) in file.items.iter().enumerate() {
            if plan.query().matches(item) {
                if self.affected_items.contains(&idx) {
                    return Err(EditorError::Conflict(format!(
                        "Item at index {} already targeted by previous mutation",
                        idx
                    )));
                }
                self.affected_items.insert(idx);
            }
        }
        
        Ok(())
    }
    
    /// Clear all tracked conflicts
    pub fn reset(&mut self) {
        self.affected_items.clear();
    }
}

impl Default for ConflictDetector {
    fn default() -> Self {
        Self::new()
    }
}
