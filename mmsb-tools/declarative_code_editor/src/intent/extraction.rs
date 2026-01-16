use super::category::EditIntent;
use crate::error::EditorError;
use syn::{File, Item, ItemFn, ItemStruct, ItemEnum, ItemUse, ItemImpl};
use std::collections::HashMap;

/// Symbol identity for tracking items across AST changes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SymbolIdentity {
    kind: SymbolKind,
    name: String,
    /// For impl blocks, the type being implemented
    impl_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SymbolKind {
    Function,
    Struct,
    Enum,
    Use,
    ImplMethod,
    #[allow(dead_code)]
    Other,
}

/// Symbol table for an AST
#[derive(Debug)]
struct SymbolTable {
    /// All symbols indexed by identity
    symbols: HashMap<SymbolIdentity, SymbolInfo>,
}

#[derive(Debug, Clone)]
struct SymbolInfo {
    #[allow(dead_code)]
    identity: SymbolIdentity,
    signature: String,
    #[allow(dead_code)]
    visibility: String,
    item_index: usize,
}

/// Extract all intents by diffing two complete ASTs
pub fn extract_intents_from_asts(
    before: &File,
    after: &File,
) -> Result<Vec<EditIntent>, EditorError> {
    let before_table = build_symbol_table(before);
    let after_table = build_symbol_table(after);
    
    let mut intents = Vec::new();
    
    // Detect renames, deletions, and modifications
    detect_changes(&before_table, &after_table, &mut intents);
    
    // Detect additions
    detect_additions(&before_table, &after_table, &mut intents);
    
    Ok(intents)
}

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

fn build_symbol_table(ast: &File) -> SymbolTable {
    let mut symbols = HashMap::new();
    
    for (idx, item) in ast.items.iter().enumerate() {
        match item {
            Item::Fn(item_fn) => {
                let identity = SymbolIdentity {
                    kind: SymbolKind::Function,
                    name: item_fn.sig.ident.to_string(),
                    impl_type: None,
                };
                let info = SymbolInfo {
                    identity: identity.clone(),
                    signature: format_fn_signature(&item_fn.sig),
                    visibility: format_visibility(&item_fn.vis),
                    item_index: idx,
                };
                symbols.insert(identity, info);
            }
            Item::Struct(item_struct) => {
                let identity = SymbolIdentity {
                    kind: SymbolKind::Struct,
                    name: item_struct.ident.to_string(),
                    impl_type: None,
                };
                let info = SymbolInfo {
                    identity: identity.clone(),
                    signature: item_struct.ident.to_string(),
                    visibility: format_visibility(&item_struct.vis),
                    item_index: idx,
                };
                symbols.insert(identity, info);
            }
            Item::Enum(item_enum) => {
                let identity = SymbolIdentity {
                    kind: SymbolKind::Enum,
                    name: item_enum.ident.to_string(),
                    impl_type: None,
                };
                let info = SymbolInfo {
                    identity: identity.clone(),
                    signature: item_enum.ident.to_string(),
                    visibility: format_visibility(&item_enum.vis),
                    item_index: idx,
                };
                symbols.insert(identity, info);
            }
            Item::Use(item_use) => {
                let path_str = quote::quote!(#item_use).to_string();
                let identity = SymbolIdentity {
                    kind: SymbolKind::Use,
                    name: path_str.clone(),
                    impl_type: None,
                };
                let info = SymbolInfo {
                    identity: identity.clone(),
                    signature: path_str,
                    visibility: format_visibility(&item_use.vis),
                    item_index: idx,
                };
                symbols.insert(identity, info);
            }
            Item::Impl(item_impl) => {
                let impl_type = extract_impl_type(item_impl);
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Fn(method) = impl_item {
                        let identity = SymbolIdentity {
                            kind: SymbolKind::ImplMethod,
                            name: method.sig.ident.to_string(),
                            impl_type: Some(impl_type.clone()),
                        };
                        let info = SymbolInfo {
                            identity: identity.clone(),
                            signature: format_fn_signature(&method.sig),
                            visibility: format_visibility(&method.vis),
                            item_index: idx,
                        };
                        symbols.insert(identity, info);
                    }
                }
            }
            _ => {}
        }
    }
    
    SymbolTable { symbols }
}

fn detect_changes(
    before: &SymbolTable,
    after: &SymbolTable,
    intents: &mut Vec<EditIntent>,
) {
    for (before_id, before_info) in &before.symbols {
        if let Some(after_info) = after.symbols.get(before_id) {
            // Symbol exists in both - check for modifications
            if before_info.signature != after_info.signature {
                intents.push(EditIntent::SignatureChange {
                    name: before_id.name.clone(),
                });
            }
        } else {
            // Symbol deleted or renamed
            // Try to find it by position in after table (indicates rename)
            let mut found_rename = false;
            for (after_id, after_info) in &after.symbols {
                if before_id.kind == after_id.kind 
                    && before_info.item_index == after_info.item_index
                    && before_id.name != after_id.name {
                    // Same position, same kind, different name = rename
                    intents.push(EditIntent::RenameSymbol {
                        old: before_id.name.clone(),
                        new: after_id.name.clone(),
                    });
                    found_rename = true;
                    break;
                }
            }
            
            if !found_rename {
                // True deletion
                intents.push(EditIntent::DeleteSymbol {
                    name: before_id.name.clone(),
                });
            }
        }
    }
}

fn detect_additions(
    before: &SymbolTable,
    after: &SymbolTable,
    intents: &mut Vec<EditIntent>,
) {
    for (after_id, _) in &after.symbols {
        if !before.symbols.contains_key(after_id) {
            // New symbol added (unless it was a rename)
            let is_rename = intents.iter().any(|intent| {
                matches!(intent, EditIntent::RenameSymbol { new, .. } if new == &after_id.name)
            });
            
            if !is_rename {
                let kind_str = match after_id.kind {
                    SymbolKind::Function => "function",
                    SymbolKind::Struct => "struct",
                    SymbolKind::Enum => "enum",
                    SymbolKind::Use => "import",
                    SymbolKind::ImplMethod => "method",
                    SymbolKind::Other => "item",
                };
                
                if after_id.kind == SymbolKind::Use {
                    intents.push(EditIntent::ImportChange {
                        path: after_id.name.clone(),
                        added: true,
                    });
                } else {
                    intents.push(EditIntent::AddSymbol {
                        name: after_id.name.clone(),
                        kind: kind_str.to_string(),
                    });
                }
            }
        }
    }
}

fn format_fn_signature(sig: &syn::Signature) -> String {
    quote::quote!(#sig).to_string()
}

fn format_visibility(vis: &syn::Visibility) -> String {
    quote::quote!(#vis).to_string()
}

fn extract_impl_type(impl_block: &ItemImpl) -> String {
    let self_ty = &impl_block.self_ty;
    quote::quote!(#self_ty).to_string()
}
