//! EditIntent translation
//!
//! Translates declarative_code_editor::EditIntent
//! to structural_code_editor::EditIntent for propagation.

use crate::intent::category::EditIntent as DeclarativeIntent;

/// structural_code_editor's EditIntent (simplified for propagation)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralIntent {
    RenameSymbol { old: String, new: String },
    DeleteSymbol { name: String },
    AddSymbol { name: String },
    SignatureChange { name: String },
}

/// Translate declarative intent to structural intent
pub fn translate_intent(intent: &DeclarativeIntent) -> Option<StructuralIntent> {
    match intent {
        DeclarativeIntent::RenameSymbol { old, new } => {
            Some(StructuralIntent::RenameSymbol {
                old: old.clone(),
                new: new.clone(),
            })
        }
        DeclarativeIntent::DeleteSymbol { name } => {
            Some(StructuralIntent::DeleteSymbol {
                name: name.clone(),
            })
        }
        DeclarativeIntent::AddSymbol { name, .. } => {
            Some(StructuralIntent::AddSymbol {
                name: name.clone(),
            })
        }
        DeclarativeIntent::SignatureChange { name } => {
            Some(StructuralIntent::SignatureChange {
                name: name.clone(),
            })
        }
        // Import and module changes don't propagate to other files
        DeclarativeIntent::ImportChange { .. } => None,
        DeclarativeIntent::ModuleChange { .. } => None,
    }
}

/// Translate multiple intents
pub fn translate_intents(intents: &[DeclarativeIntent]) -> Vec<StructuralIntent> {
    intents
        .iter()
        .filter_map(translate_intent)
        .collect()
}
