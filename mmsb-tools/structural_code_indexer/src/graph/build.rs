//! Dependency Graph Construction
//!
//! Builds DependencyGraph from extracted references.

use crate::extract::{ExtractedRef, RefType};
use crate::index::{PageIndex, SymbolIndex};
use crate::graph::edge_types::ref_type_to_edge_type;
use mmsb_core::dag::{DependencyGraph, StructuralOp};
use std::collections::HashSet;

/// Build dependency graph from extracted references
///
/// **Read-only:** Does not modify source files
/// **Possibly cyclic:** Validation happens later
pub fn build_dependency_graph(
    refs: &[ExtractedRef],
    page_index: &mut PageIndex,
    symbol_index: &mut SymbolIndex,
) -> DependencyGraph {
    let mut graph = DependencyGraph::new();
    let mut edges_to_add = HashSet::new();
    
    // First pass: register all symbols
    for extracted_ref in refs {
        let source_page = page_index.get_or_create_page(&extracted_ref.source_path);
        
        // For module references, try to resolve to target page
        if extracted_ref.ref_type == RefType::Module {
            if let Some(target_path) = symbol_index.resolve_mod_to_path(
                &extracted_ref.source_path,
                &extracted_ref.target_symbol,
            ) {
                let target_page = page_index.get_or_create_page(&target_path);
                symbol_index.register_symbol(extracted_ref.target_symbol.clone(), target_page);
            }
        }
    }
    
    // Second pass: build edges
    for extracted_ref in refs {
        let source_page = page_index.get_or_create_page(&extracted_ref.source_path);
        
        // Try to resolve symbol to target page
        let target_page = match extracted_ref.ref_type {
            RefType::Module => {
                // Module reference - resolve through symbol index
                if let Some(target_path) = symbol_index.resolve_mod_to_path(
                    &extracted_ref.source_path,
                    &extracted_ref.target_symbol,
                ) {
                    Some(page_index.get_or_create_page(&target_path))
                } else {
                    None
                }
            }
            RefType::Import | RefType::Call => {
                // Try symbol resolution
                symbol_index.resolve_symbol(&extracted_ref.target_symbol)
                    .or_else(|| {
                        // Fallback: extract crate name and create pseudo-page
                        extract_crate_from_symbol(&extracted_ref.target_symbol)
                            .map(|crate_name| {
                                // Create stable ID for external crate
                                page_index.get_or_create_page(
                                    &std::path::PathBuf::from(format!("external/{}", crate_name))
                                )
                            })
                    })
            }
        };
        
        if let Some(target) = target_page {
            let edge_type = ref_type_to_edge_type(extracted_ref.ref_type);
            edges_to_add.insert((source_page, target, edge_type));
        }
    }
    
    // Add all edges to graph
    let ops: Vec<StructuralOp> = edges_to_add
        .into_iter()
        .map(|(from, to, edge_type)| StructuralOp::AddEdge { from, to, edge_type })
        .collect();
    
    graph.apply_ops(&ops);
    
    graph
}

/// Extract crate name from symbol path
///
/// Example: "std::collections::HashMap" → "std"
fn extract_crate_from_symbol(symbol: &str) -> Option<String> {
    symbol.split("::").next().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_crate() {
        assert_eq!(
            extract_crate_from_symbol("std::collections::HashMap"),
            Some("std".to_string())
        );
        assert_eq!(
            extract_crate_from_symbol("serde"),
            Some("serde".to_string())
        );
    }
}
