//! Propagation orchestration
//!
//! Calls structural_code_editor's propagation engine and returns derived deltas.

use crate::error::EditorError;
use crate::intent::category::EditIntent;
use crate::propagation::conversion::{source_buffer_to_page_index, PageIndex};
use crate::propagation::intent_translator::{translate_intents, StructuralIntent};
use crate::source::SourceBuffer;
use mmsb_core::dag::DependencyGraph;
use mmsb_core::prelude::Delta;
use mmsb_core::types::{DeltaID, Epoch, PageID, Source};
use mmsb_judgment::JudgmentToken;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of propagation
#[derive(Debug, Clone)]
pub struct PropagatedDelta {
    pub page_id: PageID,
    pub delta: Delta,
    pub reason: String,
}

/// Propagate edits from a root page through the DAG
///
/// # Arguments
/// * `root_buffer` - The source buffer that was edited
/// * `intents` - The semantic intents extracted from the edit
/// * `graph` - The dependency graph
/// * `source_store` - Map of PageID → source code for all pages
/// * `judgment` - Judgment token for authorization
///
/// # Returns
/// Vector of derived deltas for pages affected by the propagation
pub fn propagate_from_buffer(
    root_buffer: &SourceBuffer,
    intents: &[EditIntent],
    graph: &DependencyGraph,
    source_store: &HashMap<PageID, String>,
    judgment: &JudgmentToken,
) -> Result<Vec<PropagatedDelta>, EditorError> {
    // Convert intents to structural format
    let structural_intents = translate_intents(intents);
    if structural_intents.is_empty() {
        return Ok(Vec::new());
    }
    
    // Build PageIndex for root
    let root_index = source_buffer_to_page_index(root_buffer);
    let root_page = root_index.page_id;
    
    // Build index store for all pages in source_store
    let mut index_store = HashMap::new();
    index_store.insert(root_page, root_index);
    
    for (page_id, source) in source_store {
        if *page_id == root_page {
            continue; // Already indexed
        }
        let index = index_page_from_source(*page_id, source);
        index_store.insert(*page_id, index);
    }
    
    // Propagate edits through the DAG
    let results = propagate_edits(
        root_page,
        &structural_intents,
        graph,
        &index_store,
        source_store,
        judgment,
    );
    
    Ok(results)
}

/// Build PageIndex from source code and PageID
fn index_page_from_source(page_id: PageID, src: &str) -> PageIndex {
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
    
    use syn::visit::Visit;
   use syn::{Item, ItemConst, ItemEnum, ItemFn, ItemStruct, ItemType, ItemUse, UseTree};
    
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
    
    let mut visitor = ReferenceVisitor::new(&mut references);
    visitor.visit_file(&file);
    
    PageIndex {
        page_id,
        exports,
        imports,
        references,
    }
}

fn collect_use_paths(item: &ItemUse, imports: &mut HashSet<String>) {
    let mut paths = Vec::new();
    use_tree_paths(&item.tree, String::new(), &mut paths);
    for path in paths {
        imports.insert(path);
    }
}

fn use_tree_paths(tree: &UseTree, prefix: String, out: &mut Vec<String>) {
    use syn::UseTree;
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

/// Core propagation logic (matches structural_code_editor)
fn propagate_edits(
    root_page: PageID,
    intents: &[StructuralIntent],
    graph: &DependencyGraph,
    index_store: &HashMap<PageID, PageIndex>,
    source_store: &HashMap<PageID, String>,
    judgment: &JudgmentToken,
) -> Vec<PropagatedDelta> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut epoch = 0u32;
    
    queue.push_back(root_page);
    visited.insert(root_page);
    
    while let Some(current) = queue.pop_front() {
        let Some(edges) = graph.get_adjacency().get(&current) else {
            continue;
        };
        
        for (neighbor, _) in edges {
            if visited.insert(*neighbor) {
                queue.push_back(*neighbor);
            }
            
            let Some(index) = index_store.get(neighbor) else {
                continue;
            };
            let Some(src) = source_store.get(neighbor) else {
                continue;
            };
            
            if !matches_intents(index, intents) {
                continue;
            }
            
            if let Some(new_src) = rewrite_page(src, intents) {
                let reason = format!("propagated from {}", root_page.0);
                let delta = build_delta(*neighbor, &new_src, epoch, &reason);
                epoch = epoch.wrapping_add(1);
                let _ = judgment;
                results.push(PropagatedDelta {
                    page_id: *neighbor,
                    delta,
                    reason,
                });
            }
        }
    }
    
    results
}

fn matches_intents(index: &PageIndex, intents: &[StructuralIntent]) -> bool {
    intents.iter().any(|intent| match intent {
        StructuralIntent::RenameSymbol { old, .. } => {
            index.imports.contains(old) || index.references.contains(old)
        }
        StructuralIntent::DeleteSymbol { name } => {
            index.imports.contains(name) || index.references.contains(name)
        }
        StructuralIntent::AddSymbol { .. } => false,
        StructuralIntent::SignatureChange { name } => {
            index.imports.contains(name) || index.references.contains(name)
        }
    })
}

fn rewrite_page(src: &str, intents: &[StructuralIntent]) -> Option<String> {
    let mut updated = src.to_string();
    let mut changed = false;
    
    for intent in intents {
        if let StructuralIntent::RenameSymbol { old, new } = intent {
            let (next, did_change) = replace_identifiers(&updated, old, new);
            if did_change {
                changed = true;
                updated = next;
            }
        }
    }
    
    for intent in intents {
        if let StructuralIntent::DeleteSymbol { name } = intent {
            let (next, did_change) = comment_out_usage(&updated, name);
            if did_change {
                changed = true;
                updated = next;
            }
        }
    }
    
    if changed {
        Some(updated)
    } else {
        None
    }
}

fn replace_identifiers(src: &str, old: &str, new: &str) -> (String, bool) {
    let mut out = String::with_capacity(src.len());
    let mut changed = false;
    let bytes = src.as_bytes();
    let mut idx = 0usize;
    
    while idx < bytes.len() {
        if is_ident_start(bytes[idx]) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_ident_continue(bytes[idx]) {
                idx += 1;
            }
            let token = &src[start..idx];
            if token == old {
                out.push_str(new);
                changed = true;
            } else {
                out.push_str(token);
            }
            continue;
        }
        
        out.push(bytes[idx] as char);
        idx += 1;
    }
    
    (out, changed)
}

fn comment_out_usage(src: &str, name: &str) -> (String, bool) {
    let mut changed = false;
    let mut out = String::new();
    
    for line in src.lines() {
        if line.trim_start().starts_with("//") {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        if line_contains_ident(line, name) {
            out.push_str("// MMSB-BROKEN ");
            out.push_str(line);
            out.push('\n');
            changed = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    
    if src.ends_with('\n') {
        (out, changed)
    } else {
        (out.trim_end_matches('\n').to_string(), changed)
    }
}

fn line_contains_ident(line: &str, ident: &str) -> bool {
    let bytes = line.as_bytes();
    let mut idx = 0usize;
    
    while idx < bytes.len() {
        if is_ident_start(bytes[idx]) {
            let start = idx;
            idx += 1;
            while idx < bytes.len() && is_ident_continue(bytes[idx]) {
                idx += 1;
            }
            if &line[start..idx] == ident {
                return true;
            }
            continue;
        }
        idx += 1;
    }
    
    false
}

fn is_ident_start(byte: u8) -> bool {
    byte == b'_' || (byte as char).is_ascii_alphabetic()
}

fn is_ident_continue(byte: u8) -> bool {
    byte == b'_' || (byte as char).is_ascii_alphanumeric()
}

fn build_delta(page_id: PageID, src: &str, epoch: u32, reason: &str) -> Delta {
    let payload = src.as_bytes().to_vec();
    let mask = vec![true; payload.len()];
    let delta_id = DeltaID(hash_u64(&page_id.0.to_le_bytes(), &payload));
    
    Delta {
        delta_id,
        page_id,
        epoch: Epoch(epoch),
        mask,
        payload,
        is_sparse: false,
        timestamp: epoch as u64,
        source: Source(reason.to_string()),
        intent_metadata: None,
    }
}

fn hash_u64(prefix: &[u8], payload: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(payload);
    let hash = hasher.finalize();
    u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ])
}
use syn::{ItemUse, UseTree};
