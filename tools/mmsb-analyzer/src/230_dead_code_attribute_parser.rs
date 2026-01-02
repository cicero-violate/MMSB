#![allow(dead_code)]
/// Attribute parsing for dead code intent markers.
use std::collections::{HashMap, HashSet};
use std::path::Path;
use syn::{Attribute, Item};

use crate::dead_code_doc_comment_parser::{item_attrs, item_name, merge_doc_intent};
use crate::dead_code_intent::{
    check_planned_directory, collect_symbols, merge_intent_sources, planned_directory_intent,
    DeadCodePolicy,
};
use crate::dead_code_test_boundaries::has_test_attr;
use crate::dead_code_types::{IntentMap, IntentMarker, IntentMetadata, IntentSource, IntentTag};

pub fn parse_mmsb_latent_attr(path: &Path) -> HashMap<String, Vec<IntentMetadata>> {
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    let file = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return HashMap::new(),
    };
    let mut map: HashMap<String, Vec<IntentMetadata>> = HashMap::new();
    for item in &file.items {
        let Some(name) = item_name(item) else {
            continue;
        };
        let tags = collect_latent_attrs(item_attrs(item));
        if tags.is_empty() {
            continue;
        }
        map.entry(name).or_default().extend(tags);
    }
    map
}

pub fn scan_file_attributes(path: &Path) -> Vec<IntentTag> {
    let contents = std::fs::read_to_string(path).unwrap_or_default();
    let file = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };
    let mut tags = Vec::new();
    for item in &file.items {
        let Some(symbol) = item_name(item) else {
            continue;
        };
        for meta in collect_latent_attrs(item_attrs(item)) {
            tags.push(IntentTag {
                symbol: symbol.clone(),
                file: path.to_path_buf(),
                line: None,
                marker: meta.marker,
                source: meta.source,
                value: meta.value.clone(),
            });
        }
    }
    tags
}

pub fn extract_attribute_value(attr: &Attribute, key: &str) -> Option<String> {
    let mut found = None;
    let _ = attr.parse_nested_meta(|meta| {
        if meta.path.is_ident(key) {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            found = Some(lit.value());
        }
        Ok(())
    });
    found
}

fn collect_latent_attrs(attrs: &[Attribute]) -> Vec<IntentMetadata> {
    let mut markers = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("mmsb_latent") {
            continue;
        }
        let mut marker = IntentMarker::Latent;
        let mut value = None;
        let mut saw_nested = false;
        let _ = attr.parse_nested_meta(|meta| {
            saw_nested = true;
            if meta.path.is_ident("planned") {
                marker = IntentMarker::Planned;
            } else if meta.path.is_ident("future") {
                marker = IntentMarker::Future;
            } else if meta.path.is_ident("deprecated_planned")
                || meta.path.is_ident("deprecated-planned")
            {
                marker = IntentMarker::DeprecatedPlanned;
            } else if meta.path.is_ident("reason") || meta.path.is_ident("note") {
                let value_meta = meta.value()?;
                let lit: syn::LitStr = value_meta.parse()?;
                value = Some(lit.value());
            } else if meta.path.is_ident("marker") {
                let value_meta = meta.value()?;
                let lit: syn::LitStr = value_meta.parse()?;
                marker = marker_from_str(&lit.value());
            }
            Ok(())
        });
        if !saw_nested {
            if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                value = Some(lit.value());
            }
        }

        markers.push(IntentMetadata {
            marker,
            source: IntentSource::Attribute,
            value,
        });
    }
    markers
}

fn marker_from_str(raw: &str) -> IntentMarker {
    match raw.to_ascii_lowercase().as_str() {
        "planned" => IntentMarker::Planned,
        "future" => IntentMarker::Future,
        "deprecated_planned" | "deprecated-planned" => IntentMarker::DeprecatedPlanned,
        _ => IntentMarker::Latent,
    }
}

pub fn scan_intent_tags(file: &Path, policy: Option<&DeadCodePolicy>) -> Vec<IntentTag> {
    let mut tags = Vec::new();
    let attrs = parse_mmsb_latent_attr(file);
    for (symbol, items) in attrs {
        for meta in items {
            tags.push(IntentTag {
                symbol: symbol.clone(),
                file: file.to_path_buf(),
                line: None,
                marker: meta.marker,
                source: meta.source,
                value: meta.value.clone(),
            });
        }
    }

    let doc_map = scan_doc_comments(file);
    for (symbol, markers) in doc_map {
        for marker in markers {
            tags.push(IntentTag {
                symbol: symbol.clone(),
                file: file.to_path_buf(),
                line: None,
                marker,
                source: IntentSource::DocComment,
                value: None,
            });
        }
    }

    if check_planned_directory(file, policy) {
        for symbol in collect_symbols(file) {
            tags.push(IntentTag {
                symbol,
                file: file.to_path_buf(),
                line: None,
                marker: IntentMarker::Planned,
                source: IntentSource::Directory,
                value: None,
            });
        }
    }

    tags
}



pub fn detect_intent_signals(file: &Path, policy: Option<&DeadCodePolicy>) -> IntentMap {
    let attrs = parse_mmsb_latent_attr(file);
    let doc_map = scan_doc_comments(file);
    let docs = merge_doc_intent(doc_map);
    let dir_map = planned_directory_intent(file, policy);
    merge_intent_sources(attrs, docs, dir_map)
}

#[path = "030_is_cfg_test_item.rs"]
mod moved_is_cfg_test_item;
pub use moved_is_cfg_test_item::is_cfg_test_item;

pub fn detect_test_modules(file: &Path) -> HashSet<String> {
    let contents = std::fs::read_to_string(file).unwrap_or_default();
    let parsed = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return HashSet::new(),
    };
    let mut modules = HashSet::new();
    for item in &parsed.items {
        if let Item::Mod(item_mod) = item {
            if is_cfg_test_item(item) {
                modules.insert(item_mod.ident.to_string());
            }
        }
    }
    modules
}

pub fn detect_test_symbols(file: &Path) -> HashSet<String> {
    let contents = std::fs::read_to_string(file).unwrap_or_default();
    let parsed = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return HashSet::new(),
    };
    let mut symbols = HashSet::new();
    for item in &parsed.items {
        if let Item::Fn(item_fn) = item {
            if has_test_attr(&item_fn.attrs) {
                symbols.insert(item_fn.sig.ident.to_string());
            }
        }
        if let Item::Mod(item_mod) = item {
            if is_cfg_test_item(item) {
                symbols.insert(item_mod.ident.to_string());
                if let Some((_, items)) = &item_mod.content {
                    for nested in items {
                        if let Item::Fn(nested_fn) = nested {
                            symbols.insert(nested_fn.sig.ident.to_string());
                        }
                    }
                }
            }
        }
    }
    symbols
}

#[path = "211_dead_code_doc_comment_scanner.rs"]
mod moved_scan_doc_comments;
pub use moved_scan_doc_comments::scan_doc_comments;
