//! Rust Language Extraction
//!
//! Parses Rust files and extracts references.

use std::path::PathBuf;
use syn::{File, Item, ItemMod, ItemUse, UseTree};

/// Extracted reference from source code
#[derive(Debug, Clone)]
pub struct ExtractedRef {
    pub source_path: PathBuf,
    pub target_symbol: String,
    pub ref_type: RefType,
}

/// Type of reference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefType {
    Import,
    Module,
    Call,
}

/// Extract references from Rust source code
///
/// **Over-approximation is acceptable** - we prefer false positives to false negatives.
pub fn extract_rust_refs(source_path: PathBuf, content: &str) -> Vec<ExtractedRef> {
    let mut refs = Vec::new();
    
    // Parse with syn
    let Ok(syntax_tree) = syn::parse_file(content) else {
        // Failed to parse - return empty
        return refs;
    };
    
    extract_from_file(&source_path, &syntax_tree, &mut refs);
    
    refs
}

fn extract_from_file(source_path: &PathBuf, file: &File, refs: &mut Vec<ExtractedRef>) {
    for item in &file.items {
        extract_from_item(source_path, item, refs);
    }
}

fn extract_from_item(source_path: &PathBuf, item: &Item, refs: &mut Vec<ExtractedRef>) {
    match item {
        Item::Use(use_item) => {
            extract_from_use(source_path, use_item, refs);
        }
        Item::Mod(mod_item) => {
            extract_from_mod(source_path, mod_item, refs);
        }
        _ => {}
    }
}

fn extract_from_use(source_path: &PathBuf, use_item: &ItemUse, refs: &mut Vec<ExtractedRef>) {
    extract_use_tree(source_path, &use_item.tree, String::new(), refs);
}

fn extract_use_tree(
    source_path: &PathBuf,
    tree: &UseTree,
    prefix: String,
    refs: &mut Vec<ExtractedRef>,
) {
    match tree {
        UseTree::Path(path) => {
            let new_prefix = if prefix.is_empty() {
                path.ident.to_string()
            } else {
                format!("{}::{}", prefix, path.ident)
            };
            extract_use_tree(source_path, &path.tree, new_prefix, refs);
        }
        UseTree::Name(name) => {
            let symbol = if prefix.is_empty() {
                name.ident.to_string()
            } else {
                format!("{}::{}", prefix, name.ident)
            };
            refs.push(ExtractedRef {
                source_path: source_path.clone(),
                target_symbol: symbol,
                ref_type: RefType::Import,
            });
        }
        UseTree::Rename(rename) => {
            let symbol = if prefix.is_empty() {
                rename.ident.to_string()
            } else {
                format!("{}::{}", prefix, rename.ident)
            };
            refs.push(ExtractedRef {
                source_path: source_path.clone(),
                target_symbol: symbol,
                ref_type: RefType::Import,
            });
        }
        UseTree::Glob(_) => {
            // Glob import - record the prefix
            if !prefix.is_empty() {
                refs.push(ExtractedRef {
                    source_path: source_path.clone(),
                    target_symbol: format!("{}::*", prefix),
                    ref_type: RefType::Import,
                });
            }
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                extract_use_tree(source_path, tree, prefix.clone(), refs);
            }
        }
    }
}

fn extract_from_mod(source_path: &PathBuf, mod_item: &ItemMod, refs: &mut Vec<ExtractedRef>) {
    refs.push(ExtractedRef {
        source_path: source_path.clone(),
        target_symbol: mod_item.ident.to_string(),
        ref_type: RefType::Module,
    });
    
    // If inline module, recursively extract
    if let Some((_, items)) = &mod_item.content {
        for item in items {
            extract_from_item(source_path, item, refs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    
    #[test]
    fn test_extracts_use_import() {
        let code = "use std::collections::HashMap;";
        let refs = extract_rust_refs(PathBuf::from("test.rs"), code);
        
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].target_symbol, "std::collections::HashMap");
        assert_eq!(refs[0].ref_type, RefType::Import);
    }
    
    #[test]
    fn test_extracts_mod() {
        let code = "mod foo;";
        let refs = extract_rust_refs(PathBuf::from("test.rs"), code);
        
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].target_symbol, "foo");
        assert_eq!(refs[0].ref_type, RefType::Module);
    }
    
    #[test]
    fn test_handles_parse_error() {
        let code = "not valid rust @#$%";
        let refs = extract_rust_refs(PathBuf::from("test.rs"), code);
        
        assert_eq!(refs.len(), 0);
    }
}
