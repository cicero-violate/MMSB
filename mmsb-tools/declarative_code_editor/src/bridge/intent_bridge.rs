use crate::buffer::EditBuffer;
use crate::error::EditorError;
use crate::intent::EditIntent;
use crate::executor::mutation_planner::PlannedEdit;
use syn::Item;

/// Extract semantic intent from declarative mutations
pub struct IntentBridge;

impl IntentBridge {
    /// Extract intent from planned edits
    pub fn extract_from_planned(
        planned: &[PlannedEdit],
        buffer: &EditBuffer,
    ) -> Result<Vec<EditIntent>, EditorError> {
        let mut intents = Vec::new();
        
        for plan in planned {
            let intent = Self::analyze_edit(plan, buffer)?;
            intents.extend(intent);
        }
        
        Ok(intents)
    }
    
    /// Analyze single planned edit for semantic intent
    fn analyze_edit(
        plan: &PlannedEdit,
        buffer: &EditBuffer,
    ) -> Result<Vec<EditIntent>, EditorError> {
        let mut intents = Vec::new();
        
        // Parse before and after states
        let before_items = buffer.tree().items.clone();
        
        // Find matching item by name
        let before_item = before_items.iter()
            .find(|item| extract_item_name(item) == plan.item_name);
        
        if let Some(before) = before_item {
            // Parse the new text to get after state
            if let Ok(after) = syn::parse_str::<Item>(&plan.edit.new_text) {
                let extracted = crate::intent::extraction::extract_intent(before, &after)?;
                intents.extend(extracted);
            } else {
                // If can't parse, treat as generic modification
                intents.push(EditIntent::SignatureChange {
                    name: plan.item_name.clone(),
                });
            }
        }
        
        Ok(intents)
    }
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
