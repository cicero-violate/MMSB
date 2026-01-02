//! Cluster 010: Module resolution and dependency extraction utilities
//!
//! This module contains foundational functions for:
//! - Module name resolution and path mapping
//! - Rust dependency extraction from source files
//! - Julia dependency extraction from source files
//!
//! Functions moved from src/000_dependency.rs as part of Phase 2, Batch 6 refactoring.

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};

use crate::layer_utilities::resolve_source_root;

pub use crate::cluster_001::order_julia_files_by_dependency;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use syn::ItemUse;
use walkdir::WalkDir;

use crate::dependency::RootState;

// ============================================================================
// Module Resolution (from src/000_dependency.rs)
// ============================================================================

pub fn normalize_module_name(name: &str) -> String {
    if let Some(pos) = name.find('_') {
        if name[..pos].chars().all(|c| c.is_ascii_digit()) {
            return name[pos + 1..].to_string();
        }
    }
    name.to_string()
}

pub fn resolve_module(
    root: &str,
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    let key = normalize_module_name(root);
    if let Some(path) = module_map.get(&key) {
        return Some(path.clone());
    }
    module_map
        .iter()
        .find(|(name, _)| name == &&key)
        .map(|(_, path)| path.clone())
        .or_else(|| {
            module_map
                .iter()
                .find(|(name, _)| key.starts_with(name.as_str()))
                .map(|(_, path)| path.clone())
        })
        .or_else(|| crate::cluster_011::resolve_path(&PathBuf::from(root), file_set, module_map))
}

pub fn contains_tools(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == "tools")
}

#[derive(Clone)]
pub struct ModuleRoot {
    pub layer: String,
}

pub fn build_module_root_map(root: &Path) -> Result<HashMap<String, ModuleRoot>, std::io::Error> {
    let src_dir = root.join("src");
    let mut map = HashMap::new();
    if src_dir.is_dir() {
        for entry in fs::read_dir(&src_dir)? {
            let entry = entry?;
            let path = entry.path();
            if contains_tools(&path) {
                continue;
            }
            let name = entry
                .file_name()
                .to_string_lossy()
                .to_string()
                .trim_end_matches(".rs")
                .to_string();
            if path.is_dir() {
                let normalized = normalize_module_name(&name);
                map.insert(
                    normalized,
                    ModuleRoot {
                        layer: name.clone(),
                    },
                );
            } else if path.extension().map(|ext| ext == "rs").unwrap_or(false) {
                map.insert(
                    name.clone(),
                    ModuleRoot {
                        layer: crate::cluster_001::detect_layer(&path),
                    },
                );
            }
        }
    }
    Ok(map)
}

// ============================================================================
// Julia Dependency Ordering (from src/020_layer_utilities.rs)
// ============================================================================

pub(crate) struct LayerResolver {
    aliases: HashMap<String, String>,
}

impl LayerResolver {
    pub(crate) fn build(root: &Path) -> Result<Self> {
        let mut resolver = LayerResolver {
            aliases: HashMap::new(),
        };
        let src_dir = resolve_source_root(root);
        if src_dir.is_dir() {
            for entry in WalkDir::new(&src_dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if contains_tools(path) {
                    continue;
                }
                let layer = crate::cluster_001::detect_layer(path);
                if layer == "root" {
                    continue;
                }
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        resolver.add_aliases(name, &layer);
                    }
                } else if path.extension().map_or(false, |ext| ext == "jl") {
                    if let Some(stem) = path.file_stem().and_then(|n| n.to_str()) {
                        resolver.add_aliases(stem, &layer);
                    }
                }
            }
        }
        Ok(resolver)
    }

    fn add_aliases(&mut self, name: &str, layer: &str) {
        let lower = name.to_lowercase();
        self.aliases
            .entry(lower.clone())
            .or_insert_with(|| layer.to_string());
        let condensed = lower.replace('_', "");
        self.aliases
            .entry(condensed)
            .or_insert_with(|| layer.to_string());
    }

    pub(crate) fn resolve_module(&self, module: &str) -> Option<String> {
        let key = module.to_lowercase();
        if let Some(layer) = self.aliases.get(&key) {
            return Some(layer.clone());
        }
        let condensed = key.replace('_', "");
        if let Some(layer) = self.aliases.get(&condensed) {
            return Some(layer.clone());
        }
        self.aliases
            .iter()
            .filter(|(alias, _)| !alias.is_empty())
            .find(|(alias, _)| key.starts_with(alias.as_str()))
            .map(|(_, layer)| layer.clone())
    }
}



// ============================================================================
// Rust Dependency Extraction (from src/000_dependency.rs)
// ============================================================================

pub fn extract_rust_dependencies(
    file: &Path,
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Result<Vec<PathBuf>> {
    #[derive(Default)]
    struct UseCollector {
        roots: BTreeSet<String>,
        mods: BTreeSet<String>,
    }

    impl<'ast> Visit<'ast> for UseCollector {
        fn visit_item_use(&mut self, node: &'ast ItemUse) {
            crate::dependency::collect_roots(&node.tree, RootState::Start, &mut self.roots);
        }

        fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
            if node.content.is_none() {
                self.mods.insert(node.ident.to_string());
            }
        }
    }

    let content =
        fs::read_to_string(file).with_context(|| format!("Unable to read {:?}", file))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Unable to parse Rust file {:?}", file))?;
    let mut collector = UseCollector::default();
    collector.visit_file(&syntax);
    let mut deps = Vec::new();
    for root in collector.roots {
        if let Some(path) = resolve_module(&root, file_set, module_map) {
            deps.push(path);
        }
    }
    for module in collector.mods {
        if let Some(path) = resolve_module(&module, file_set, module_map) {
            deps.push(path);
        }
    }
    Ok(deps)
}

// ============================================================================
// Julia Dependency Extraction (from src/000_dependency.rs)
// ============================================================================

pub fn extract_julia_dependencies(
    file: &Path,
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Result<Vec<PathBuf>> {
    static INCLUDE_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r#"include\s*\(\s*["']([^"']+)["']"#).unwrap());
    static MMSB_USING_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?m)^\s*(?:using|import)\s+MMSB\.([A-Za-z0-9_\.]+)"#).unwrap()
    });
    static MMSB_SYMBOL_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?m)^\s*(?:using|import)\s+MMSB\s*:\s*([A-Za-z0-9_,\s]+)"#).unwrap()
    });
    static LOCAL_USING_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?m)^\s*(?:using|import)\s+\.\s*([A-Za-z0-9_\.]+)"#).unwrap()
    });
    static PLAIN_USING_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?m)^\s*(?:using|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)"#).unwrap()
    });

    fn resolve_module_name(
        module: &str,
        file_set: &HashSet<PathBuf>,
        module_map: &HashMap<String, PathBuf>,
    ) -> Option<PathBuf> {
        let primary = module.split('.').next().unwrap_or(module);
        resolve_module(primary, file_set, module_map)
    }

    let content =
        fs::read_to_string(file).with_context(|| format!("Unable to read {:?}", file))?;
    let mut deps = Vec::new();

    for cap in INCLUDE_RE.captures_iter(&content) {
        if let Some(path_match) = cap.get(1) {
            let raw = path_match.as_str();
            let mut candidate = PathBuf::from(raw);
            if candidate.extension().is_none() {
                candidate.set_extension("jl");
            }
            let resolved = if candidate.is_absolute() {
                candidate
            } else {
                file.parent()
                    .map(|p| p.join(&candidate))
                    .unwrap_or(candidate)
            };
            if let Some(path) = crate::cluster_011::resolve_path(&resolved, file_set, module_map) {
                deps.push(path);
            }
        }
    }

    for cap in MMSB_USING_RE.captures_iter(&content) {
        if let Some(module_match) = cap.get(1) {
            if let Some(path) = resolve_module_name(module_match.as_str(), file_set, module_map) {
                deps.push(path);
            }
        }
    }

    for cap in MMSB_SYMBOL_RE.captures_iter(&content) {
        if let Some(symbols) = cap.get(1) {
            for symbol in symbols
                .as_str()
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                if let Some(path) = resolve_module_name(symbol, file_set, module_map) {
                    deps.push(path);
                }
            }
        }
    }

    for cap in LOCAL_USING_RE.captures_iter(&content) {
        if let Some(module_match) = cap.get(1) {
            if let Some(path) = resolve_module_name(module_match.as_str(), file_set, module_map) {
                deps.push(path);
            }
        }
    }

    for cap in PLAIN_USING_RE.captures_iter(&content) {
        if let Some(module_match) = cap.get(1) {
            let module = module_match.as_str();
            if module.starts_with("MMSB") {
                continue;
            }
            if let Some(path) = resolve_module_name(module, file_set, module_map) {
                deps.push(path);
            }
        }
    }

    Ok(deps)
}

// ============================================================================
// File Dependency Mapping (from src/090_file_ordering.rs)
// ============================================================================

pub fn build_dependency_map(
    files: &[PathBuf],
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Result<HashMap<PathBuf, Vec<PathBuf>>> {
    let mut dep_map: HashMap<PathBuf, Vec<PathBuf>> = HashMap::new();
    for file in files {
        let deps = extract_dependencies(file, file_set, module_map)
            .with_context(|| format!("Failed to extract dependencies for {:?}", file))?;
        dep_map.insert(file.clone(), deps);
    }
    Ok(dep_map)
}

pub(crate) fn extract_dependencies(
    file: &Path,
    file_set: &HashSet<PathBuf>,
    module_map: &HashMap<String, PathBuf>,
) -> Result<Vec<PathBuf>> {
    let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "rs" => extract_rust_dependencies(file, file_set, module_map),
        "jl" => extract_julia_dependencies(file, file_set, module_map),
        _ => Ok(Vec::new()),
    }
}



#[path = "020_gather_rust_files.rs"]
mod moved_gather_rust_files;
pub use moved_gather_rust_files::gather_rust_files;
