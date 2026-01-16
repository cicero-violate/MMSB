//! SourceBuffer → PageIndex conversion
//!
//! Extracts exports, imports, and references from parsed AST
//! for structural_code_editor's propagation engine.

use crate::source::SourceBuffer;
use mmsb_core::types::PageID;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use syn::visit::Visit;
use syn::{Item, ItemConst, ItemEnum, ItemFn, ItemStruct, ItemType, ItemUse, UseTree};

/// PageIndex for structural_code_editor
#[derive(Debug, Clone)]
pub struct PageIndex {
    pub page_id: PageID,
    pub exports: HashSet<String>,
    pub imports: HashSet<String>,
    pub references: HashSet<String>,
}

/// Convert SourceBuffer to PageIndex
pub fn source_buffer_to_page_index(buffer: &SourceBuffer) -> PageIndex {
    let page_id = stable_page_id(&buffer.path.to_string_lossy());
    let file = buffer.ast();
    
    let mut exports = HashSet::new();
    let mut imports = HashSet::new();
    let mut references = HashSet::new();
    
    // Extract exports (top-level items)
    for item in &file.items {
        match item {
            Item::Fn(ItemFn { sig, .. }) => {
                exports.insert(sig.ident.to_string());
            }
            Item::Struct(ItemStruct { ident, .. }) => {
                exports.insert(ident.to_string());
            }
            Item::Enum(ItemEnum { ident, .. }) => {
                exports.insert(ident.to_string());
            }
            Item::Type(ItemType { ident, .. }) => {
                exports.insert(ident.to_string());
            }
            Item::Const(ItemConst { ident, .. }) => {
                exports.insert(ident.to_string());
            }
            Item::Use(use_item) => {
                collect_use_paths(use_item, &mut imports);
            }
            _ => {}
        }
    }
    
    // Extract references (identifiers used in function bodies, etc.)
    let mut visitor = ReferenceVisitor::new(&mut references);
    visitor.visit_file(file);
    
    PageIndex {
        page_id,
        exports,
        imports,
        references,
    }
}

fn stable_page_id(path: &str) -> PageID {
    let mut hasher = Sha256::new();
    hasher.update(path.as_bytes());
    let hash = hasher.finalize();
    let id = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ]);
    PageID(id)
}

fn collect_use_paths(item: &ItemUse, imports: &mut HashSet<String>) {
    let mut paths = Vec::new();
    use_tree_paths(&item.tree, String::new(), &mut paths);
    for path in paths {
        imports.insert(path);
    }
}

fn use_tree_paths(tree: &UseTree, prefix: String, out: &mut Vec<String>) {
    match tree {
        UseTree::Path(path) => {
            let next = if prefix.is_empty() {
                path.ident.to_string()
            } else {
                format!("{}::{}", prefix, path.ident)
            };
            use_tree_paths(&path.tree, next, out);
        }
        UseTree::Name(name) => {
            let path = if prefix.is_empty() {
                name.ident.to_string()
            } else {
                format!("{}::{}", prefix, name.ident)
            };
            out.push(path);
        }
        UseTree::Rename(rename) => {
            let path = if prefix.is_empty() {
                rename.ident.to_string()
            } else {
                format!("{}::{}", prefix, rename.ident)
            };
            out.push(path);
        }
        UseTree::Glob(_) => {
            if !prefix.is_empty() {
                out.push(format!("{}::*", prefix));
            }
        }
        UseTree::Group(group) => {
            for item in &group.items {
                use_tree_paths(item, prefix.clone(), out);
            }
        }
    }
}

struct ReferenceVisitor<'a> {
    references: &'a mut HashSet<String>,
}

impl<'a> ReferenceVisitor<'a> {
    fn new(references: &'a mut HashSet<String>) -> Self {
        Self { references }
    }
}

impl<'a> Visit<'a> for ReferenceVisitor<'a> {
    fn visit_path(&mut self, path: &'a syn::Path) {
        if let Some(ident) = path.get_ident() {
            self.references.insert(ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}
