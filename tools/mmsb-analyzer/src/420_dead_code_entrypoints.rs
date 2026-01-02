#![allow(dead_code)]
//! Entrypoint and export resolution for dead code analysis.

use crate::dead_code_intent::DeadCodePolicy;
use crate::types::{CodeElement, ElementType, Visibility};
use std::collections::HashSet;
use std::path::Path;
use walkdir::WalkDir;

pub fn collect_entrypoints(
    elements: &[CodeElement],
    policy: Option<&DeadCodePolicy>,
) -> HashSet<String> {
    let mut entrypoints = HashSet::new();
    if let Some(policy) = policy {
        for symbol in &policy.entrypoint_symbols {
            entrypoints.insert(symbol.clone());
        }
    }
    for element in elements {
        if element.element_type != ElementType::Function {
            continue;
        }
        if element.name == "main" {
            entrypoints.insert(element.name.clone());
            continue;
        }
        if matches!(element.visibility, Visibility::Public) && treat_public_as_entrypoint(policy) {
            entrypoints.insert(element.name.clone());
        }
    }
    entrypoints
}

pub fn collect_exports(root: &Path) -> HashSet<String> {
    let mut exports = HashSet::new();
    let src_dir = root.join("src");
    for entry in WalkDir::new(src_dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        let parsed = match syn::parse_file(&contents) {
            Ok(file) => file,
            Err(_) => continue,
        };
        for item in parsed.items {
            match item {
                syn::Item::Use(item_use) => {
                    if matches!(item_use.vis, syn::Visibility::Public(_)) {
                        collect_use_tree_idents(&item_use.tree, &mut exports);
                    }
                }
                syn::Item::Mod(item_mod) => {
                    if matches!(item_mod.vis, syn::Visibility::Public(_)) {
                        exports.insert(item_mod.ident.to_string());
                    }
                }
                _ => {}
            }
        }
    }
    exports
}

pub fn is_public_api(symbol: &str, exports: &HashSet<String>) -> bool {
    exports.contains(symbol)
}

fn collect_use_tree_idents(tree: &syn::UseTree, exports: &mut HashSet<String>) {
    match tree {
        syn::UseTree::Name(name) => {
            exports.insert(name.ident.to_string());
        }
        syn::UseTree::Rename(rename) => {
            exports.insert(rename.rename.to_string());
        }
        syn::UseTree::Path(path) => {
            collect_use_tree_idents(&path.tree, exports);
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree_idents(item, exports);
            }
        }
        syn::UseTree::Glob(_) => {
            exports.insert("*".to_string());
        }
    }
}

fn treat_public_as_entrypoint(policy: Option<&DeadCodePolicy>) -> bool {
    policy.map(|p| p.treat_public_as_entrypoint).unwrap_or(true)
}
