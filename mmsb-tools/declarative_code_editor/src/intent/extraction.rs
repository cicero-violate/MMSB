use super::category::EditIntent;
use crate::error::EditorError;
use syn::{Item, ItemFn, ItemStruct, ItemEnum, ItemUse};

// TODO: Advanced intent extraction
// - Detect cascading renames (struct + all its methods)
// - Track visibility changes (pub → private impacts)
// - Lifetime changes (affects borrowing semantics)
// - Generic parameter changes (affects monomorphization)
// - Trait bound additions/removals (affects implementation)
// - Macro invocation changes (may expand differently)
// - Unsafe block additions (requires verification)
// - Derive changes (affects generated code)
// - Semantic analysis (type-level changes)

/// Extract semantic intent from AST comparison
pub fn extract_intent(before: &Item, after: &Item) -> Result<Vec<EditIntent>, EditorError> {
    let mut intents = Vec::new();

    match (before, after) {
        (Item::Fn(before_fn), Item::Fn(after_fn)) => {
            extract_fn_intent(before_fn, after_fn, &mut intents);
        }
        (Item::Struct(before_s), Item::Struct(after_s)) => {
            extract_struct_intent(before_s, after_s, &mut intents);
        }
        (Item::Enum(before_e), Item::Enum(after_e)) => {
            extract_enum_intent(before_e, after_e, &mut intents);
        }
        (Item::Use(before_u), Item::Use(after_u)) => {
            extract_use_intent(before_u, after_u, &mut intents);
        }
        _ => {
            // Generic replacement
        }
    }

    Ok(intents)
}

fn extract_fn_intent(before: &ItemFn, after: &ItemFn, intents: &mut Vec<EditIntent>) {
    let before_name = before.sig.ident.to_string();
    let after_name = after.sig.ident.to_string();

    if before_name != after_name {
        intents.push(EditIntent::RenameSymbol {
            old: before_name.clone(),
            new: after_name.clone(),
        });
    }

    // Check signature changes (params, return type)
    if before.sig.inputs != after.sig.inputs || before.sig.output != after.sig.output {
        intents.push(EditIntent::SignatureChange {
            name: after_name,
        });
    }
}

fn extract_struct_intent(before: &ItemStruct, after: &ItemStruct, intents: &mut Vec<EditIntent>) {
    let before_name = before.ident.to_string();
    let after_name = after.ident.to_string();

    if before_name != after_name {
        intents.push(EditIntent::RenameSymbol {
            old: before_name,
            new: after_name,
        });
    }
}

fn extract_enum_intent(before: &ItemEnum, after: &ItemEnum, intents: &mut Vec<EditIntent>) {
    let before_name = before.ident.to_string();
    let after_name = after.ident.to_string();

    if before_name != after_name {
        intents.push(EditIntent::RenameSymbol {
            old: before_name,
            new: after_name,
        });
    }
}

fn extract_use_intent(before: &ItemUse, after: &ItemUse, intents: &mut Vec<EditIntent>) {
    let before_path = quote::quote!(#before).to_string();
    let after_path = quote::quote!(#after).to_string();

    if before_path != after_path {
        intents.push(EditIntent::ImportChange {
            path: after_path,
            added: true,
        });
    }
}
