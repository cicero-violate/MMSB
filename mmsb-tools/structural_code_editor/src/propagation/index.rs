use crate::RepoSnapshot;
use mmsb_core::types::PageID;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use syn::{File, Item, ItemConst, ItemEnum, ItemFn, ItemStruct, ItemType, ItemUse, UseTree};

#[derive(Debug, Clone)]
pub struct PageIndex {
    pub page_id: PageID,
    pub exports: HashSet<String>,
    pub imports: HashSet<String>,
    pub references: HashSet<String>,
}

pub fn index_page(path: &Path, src: &str) -> PageIndex {
    let page_id = stable_page_id(path);
    let mut exports = HashSet::new();
    let mut imports = HashSet::new();
    let mut references = HashSet::new();

    let Ok(file) = syn::parse_file(src) else {
        return PageIndex {
            page_id,
            exports,
            imports,
            references,
        };
    };

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

    let mut visitor = ReferenceVisitor::new(&mut references);
    visitor.visit_file(&file);

    PageIndex {
        page_id,
        exports,
        imports,
        references,
    }
}

pub fn index_snapshot(
    snapshot: &RepoSnapshot,
) -> (HashMap<PageID, PageIndex>, HashMap<PageID, String>) {
    let mut index_store = HashMap::new();
    let mut source_store = HashMap::new();

    for file in &snapshot.files {
        let page_id = stable_page_id(&file.path);
        let index = index_page(&file.path, &file.content);
        index_store.insert(page_id, index);
        source_store.insert(page_id, file.content.clone());
    }

    (index_store, source_store)
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
            let symbol = if prefix.is_empty() {
                name.ident.to_string()
            } else {
                format!("{}::{}", prefix, name.ident)
            };
            out.push(symbol);
        }
        UseTree::Rename(rename) => {
            let symbol = if prefix.is_empty() {
                rename.ident.to_string()
            } else {
                format!("{}::{}", prefix, rename.ident)
            };
            out.push(symbol);
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

impl<'a> Visit<'_> for ReferenceVisitor<'a> {
    fn visit_item_use(&mut self, _i: &ItemUse) {
        // Imports are tracked separately.
    }

    fn visit_path(&mut self, path: &syn::Path) {
        if let Some(segment) = path.segments.last() {
            self.references.insert(segment.ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}

fn stable_page_id(path: &Path) -> PageID {
    let canonical = canonicalize_path(path);
    let text = canonical.to_string_lossy();
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    let id = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ]);
    PageID(id)
}

fn canonicalize_path(path: &Path) -> PathBuf {
    path.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir | std::path::Component::ParentDir))
        .collect()
}
