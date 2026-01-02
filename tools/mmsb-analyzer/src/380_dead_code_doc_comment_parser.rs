#![allow(dead_code)]
//! Doc comment parsing for intent markers.

use crate::dead_code_types::{IntentMarker, IntentMap};
use std::collections::{HashMap, HashSet};

use syn::{Attribute, Item, Meta, MetaNameValue};



pub fn detect_latent_markers(comment: &str) -> Option<IntentMarker> {
    IntentMarker::from_comment(comment)
}

pub fn merge_doc_intent(map: HashMap<String, Vec<IntentMarker>>) -> IntentMap {
    let mut merged = IntentMap::new();
    for (symbol, markers) in map {
        let mut uniques = HashSet::new();
        for marker in markers {
            if !uniques.insert(marker) {
                continue;
            }
            merged
                .entry(symbol.clone())
                .or_default()
                .push(crate::dead_code_types::IntentMetadata {
                    marker,
                    source: crate::dead_code_types::IntentSource::DocComment,
                    value: None,
                });
        }
    }
    merged
}

pub(crate) fn extract_doc_markers(attrs: &[Attribute]) -> Vec<IntentMarker> {
    let mut markers = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        let Meta::NameValue(MetaNameValue { value, .. }) = &attr.meta else {
            continue;
        };
        let syn::Expr::Lit(expr_lit) = value else {
            continue;
        };
        let syn::Lit::Str(lit) = &expr_lit.lit else {
            continue;
        };
        if let Some(marker) = detect_latent_markers(&lit.value()) {
            markers.push(marker);
        }
    }
    markers
}

pub(crate) fn item_name(item: &Item) -> Option<String> {
    match item {
        Item::Fn(item_fn) => Some(item_fn.sig.ident.to_string()),
        Item::Struct(item_struct) => Some(item_struct.ident.to_string()),
        Item::Enum(item_enum) => Some(item_enum.ident.to_string()),
        Item::Mod(item_mod) => Some(item_mod.ident.to_string()),
        Item::Trait(item_trait) => Some(item_trait.ident.to_string()),
        _ => None,
    }
}

pub(crate) fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Fn(item_fn) => &item_fn.attrs,
        Item::Struct(item_struct) => &item_struct.attrs,
        Item::Enum(item_enum) => &item_enum.attrs,
        Item::Mod(item_mod) => &item_mod.attrs,
        Item::Trait(item_trait) => &item_trait.attrs,
        _ => &[],
    }
}
