//! Dependency-aware ordering and layer graph extraction.

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use quote::quote;
use regex::Regex;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;
use syn::{ItemUse, UseTree};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct LayerGraph {
    pub ordered_layers: Vec<String>,
    pub edges: Vec<LayerEdge>,
    pub cycles: Vec<String>,
    pub unresolved: Vec<UnresolvedDependency>,
}

#[derive(Debug, Clone)]
pub struct LayerEdge {
    pub from: String,
    pub to: String,
    pub references: Vec<ReferenceDetail>,
    pub violation: bool,
}

#[derive(Debug, Clone)]
pub struct UnresolvedDependency {
    pub file: PathBuf,
    pub reference: String,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct ReferenceDetail {
    pub file: PathBuf,
    pub reference: String,
}

#[derive(Clone)]
struct ModuleRoot {
    layer: String,
}

struct LayerResolver {
    aliases: HashMap<String, String>,
}

impl LayerResolver {
    fn build(root: &Path) -> Result<Self> {
        let mut resolver = LayerResolver {
            aliases: HashMap::new(),
        };
        let src_dir = root.join("src");
        if src_dir.is_dir() {
            for entry in WalkDir::new(&src_dir).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if contains_tools(path) {
                    continue;
                }
                let layer = detect_layer(path);
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

    fn resolve_module(&self, module: &str) -> Option<String> {
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

#[derive(Clone)]
struct RustDependency {
    root: String,
    detail: String,
}

#[derive(Clone)]
struct JuliaDependency {
    target: JuliaTarget,
    detail: String,
}

#[derive(Clone)]
enum JuliaTarget {
    Include(PathBuf),
    Module(String),
}

/// Order Rust files by dependency and capture layer graph details.
pub fn order_rust_files_by_dependency(
    files: &[PathBuf],
    root: &Path,
) -> Result<(Vec<PathBuf>, LayerGraph)> {
    let module_map = build_module_root_map(root)?;
    let entry_files = rust_entry_paths(root);
    let mut file_layers: HashMap<PathBuf, String> = HashMap::new();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    let mut edges_map: BTreeMap<(String, String), BTreeSet<ReferenceDetail>> = BTreeMap::new();
    let mut unresolved = Vec::new();

    for file in files {
        let layer = detect_layer(file);
        nodes.insert(layer.clone());
        file_layers.insert(file.clone(), layer.clone());

        let deps = collect_rust_dependencies(file)
            .with_context(|| format!("Failed to collect dependencies for {:?}", file))?;
        for dep in deps {
            if let Some(info) = module_map.get(&dep.root) {
                nodes.insert(info.layer.clone());
                if info.layer != layer {
                    edges_map
                        .entry((info.layer.clone(), layer.clone()))
                        .or_default()
                        .insert(ReferenceDetail {
                            file: file.clone(),
                            reference: dep.detail.clone(),
                        });
                }
            } else {
                unresolved.push(UnresolvedDependency {
                    file: file.clone(),
                    reference: dep.detail.clone(),
                });
            }
        }
    }

    build_result(
        files,
        file_layers,
        nodes,
        edges_map,
        unresolved,
        &entry_files,
    )
}

/// Order Julia files by dependency references (includes/imports).
pub fn order_julia_files_by_dependency(
    files: &[PathBuf],
    root: &Path,
) -> Result<(Vec<PathBuf>, LayerGraph)> {
    let mut file_layers: HashMap<PathBuf, String> = HashMap::new();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    let mut edges_map: BTreeMap<(String, String), BTreeSet<ReferenceDetail>> = BTreeMap::new();
    let mut unresolved = Vec::new();
    let resolver = LayerResolver::build(root)?;
    let entry_files = julia_entry_paths(root);

    for file in files {
        let layer = detect_layer(file);
        nodes.insert(layer.clone());
        file_layers.insert(file.clone(), layer.clone());

        let references = collect_julia_dependencies(file)
            .with_context(|| format!("Failed to analyze Julia dependencies for {:?}", file))?;
        for dep in references {
            match dep.target {
                JuliaTarget::Include(include_path) => {
                    let resolved = if include_path.is_absolute() {
                        include_path.clone()
                    } else {
                        file.parent()
                            .map(|p| p.join(&include_path))
                            .unwrap_or(include_path.clone())
                    };

                    if resolved.exists() {
                        let target_layer = detect_layer(&resolved);
                        nodes.insert(target_layer.clone());
                        if target_layer != layer {
                            edges_map
                                .entry((target_layer.clone(), layer.clone()))
                                .or_default()
                                .insert(ReferenceDetail {
                                    file: file.clone(),
                                    reference: dep.detail.clone(),
                                });
                        }
                    } else {
                        unresolved.push(UnresolvedDependency {
                            file: file.clone(),
                            reference: dep.detail.clone(),
                        });
                    }
                }
                JuliaTarget::Module(module) => {
                    if let Some(target_layer) = resolver.resolve_module(&module) {
                        nodes.insert(target_layer.clone());
                        if target_layer != layer {
                            edges_map
                                .entry((target_layer.clone(), layer.clone()))
                                .or_default()
                                .insert(ReferenceDetail {
                                    file: file.clone(),
                                    reference: dep.detail.clone(),
                                });
                        }
                    } else {
                        unresolved.push(UnresolvedDependency {
                            file: file.clone(),
                            reference: dep.detail.clone(),
                        });
                    }
                }
            }
        }
    }

    build_result(
        files,
        file_layers,
        nodes,
        edges_map,
        unresolved,
        &entry_files,
    )
}

fn build_result(
    files: &[PathBuf],
    file_layers: HashMap<PathBuf, String>,
    nodes: BTreeSet<String>,
    edges_map: BTreeMap<(String, String), BTreeSet<ReferenceDetail>>,
    unresolved: Vec<UnresolvedDependency>,
    entry_files: &BTreeSet<PathBuf>,
) -> Result<(Vec<PathBuf>, LayerGraph)> {
    let adjacency = adjacency_from_edges(&edges_map);
    let (mut ordered_layers, cycles) = topo_sort(&nodes, &adjacency);
    if let Some(pos) = ordered_layers.iter().position(|layer| layer == "root") {
        let root_layer = ordered_layers.remove(pos);
        ordered_layers.insert(0, root_layer);
    }
    let rank = layer_rank_map(&ordered_layers);

    let mut ordered_files = files.to_vec();
    ordered_files.sort_by(|a, b| {
        let entry_a = entry_files.contains(a);
        let entry_b = entry_files.contains(b);
        if entry_a && !entry_b {
            return Ordering::Less;
        } else if entry_b && !entry_a {
            return Ordering::Greater;
        }
        let layer_a = file_layers
            .get(a)
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        let layer_b = file_layers
            .get(b)
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        let rank_a = rank.get(&layer_a).cloned().unwrap_or(ordered_layers.len());
        let rank_b = rank.get(&layer_b).cloned().unwrap_or(ordered_layers.len());
        rank_a
            .cmp(&rank_b)
            .then_with(|| layer_a.cmp(&layer_b))
            .then_with(|| a.cmp(b))
    });

    let edges = edges_map
        .into_iter()
        .map(|((from, to), references)| LayerEdge {
            violation: is_violation(&from, &to),
            from,
            to,
            references: references.into_iter().collect(),
        })
        .collect();

    let graph = LayerGraph {
        ordered_layers,
        edges,
        cycles,
        unresolved,
    };

    Ok((ordered_files, graph))
}

fn build_module_root_map(root: &Path) -> Result<HashMap<String, ModuleRoot>> {
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
                        layer: detect_layer(&path),
                    },
                );
            }
        }
    }
    Ok(map)
}

fn normalize_module_name(name: &str) -> String {
    if let Some(pos) = name.find('_') {
        if name[..pos].chars().all(|c| c.is_ascii_digit()) {
            return name[pos + 1..].to_string();
        }
    }
    name.to_string()
}

fn collect_rust_dependencies(path: &Path) -> Result<Vec<RustDependency>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Unable to read Rust file {:?}", path))?;
    let syntax = syn::parse_file(&content)
        .with_context(|| format!("Unable to parse Rust file {:?}", path))?;
    let mut collector = UseCollector::default();
    collector.visit_file(&syntax);
    Ok(collector.deps)
}

#[derive(Default)]
struct UseCollector {
    deps: Vec<RustDependency>,
}

impl<'ast> Visit<'ast> for UseCollector {
    fn visit_item_use(&mut self, node: &'ast ItemUse) {
        let mut roots = BTreeSet::new();
        collect_roots(&node.tree, ContextState::Start, &mut roots);
        let stmt = quote!(#node).to_string();
        for root in roots {
            self.deps.push(RustDependency {
                root,
                detail: stmt.clone(),
            });
        }
    }
}

fn collect_julia_dependencies(path: &Path) -> Result<Vec<JuliaDependency>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Unable to read Julia file {:?}", path))?;
    let mut deps = Vec::new();

    for cap in INCLUDE_REGEX.captures_iter(&content) {
        if let Some(path_match) = cap.get(1) {
            let relative = PathBuf::from(path_match.as_str());
            let detail = cap
                .get(0)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            deps.push(JuliaDependency {
                target: JuliaTarget::Include(relative),
                detail,
            });
        }
    }

    for cap in USING_REGEX.captures_iter(&content) {
        if let Some(module_match) = cap.get(1) {
            let module = module_match.as_str();
            let primary = module.split('.').next().unwrap_or(module).to_string();
            let detail = cap
                .get(0)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            deps.push(JuliaDependency {
                target: JuliaTarget::Module(primary),
                detail,
            });
        }
    }

    for cap in ROOT_USING_REGEX.captures_iter(&content) {
        if let Some(symbols) = cap.get(1) {
            let detail = cap
                .get(0)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            for symbol in symbols
                .as_str()
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                let primary = symbol.split('.').next().unwrap_or(symbol).to_string();
                deps.push(JuliaDependency {
                    target: JuliaTarget::Module(primary),
                    detail: detail.clone(),
                });
            }
        }
    }

    for cap in LOCAL_USING_REGEX.captures_iter(&content) {
        if let Some(module_match) = cap.get(1) {
            let module = module_match.as_str();
            let detail = cap
                .get(0)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            deps.push(JuliaDependency {
                target: JuliaTarget::Module(module.to_string()),
                detail,
            });
        }
    }

    Ok(deps)
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ContextState {
    Start,
    AfterCrate,
}

fn collect_roots(tree: &UseTree, state: ContextState, acc: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => {
            let ident = path.ident.to_string();
            if state == ContextState::Start && ident == "crate" {
                collect_roots(&path.tree, ContextState::AfterCrate, acc);
            } else if state == ContextState::AfterCrate {
                acc.insert(ident);
            } else {
                collect_roots(&path.tree, state, acc);
            }
        }
        UseTree::Group(group) => {
            for tree in &group.items {
                collect_roots(tree, state, acc);
            }
        }
        UseTree::Name(name) => {
            if state == ContextState::AfterCrate {
                acc.insert(name.ident.to_string());
            }
        }
        UseTree::Rename(rename) => {
            if state == ContextState::AfterCrate {
                acc.insert(rename.ident.to_string());
            }
        }
        UseTree::Glob(_) => {}
    }
}

fn adjacency_from_edges(
    edges_map: &BTreeMap<(String, String), BTreeSet<ReferenceDetail>>,
) -> HashMap<String, BTreeSet<String>> {
    let mut adjacency: HashMap<String, BTreeSet<String>> = HashMap::new();
    for ((from, to), _) in edges_map {
        adjacency
            .entry(from.clone())
            .or_default()
            .insert(to.clone());
    }
    adjacency
}

fn topo_sort(
    nodes: &BTreeSet<String>,
    adjacency: &HashMap<String, BTreeSet<String>>,
) -> (Vec<String>, Vec<String>) {
    let mut indegree: HashMap<String, usize> = HashMap::new();
    for node in nodes {
        indegree.entry(node.clone()).or_insert(0);
    }
    for targets in adjacency.values() {
        for target in targets {
            *indegree.entry(target.clone()).or_insert(0) += 1;
        }
    }

    let mut queue: VecDeque<String> = indegree
        .iter()
        .filter_map(|(node, &deg)| if deg == 0 { Some(node.clone()) } else { None })
        .collect();
    queue.make_contiguous().sort();

    let mut order = Vec::new();
    while let Some(node) = queue.pop_front() {
        order.push(node.clone());
        if let Some(targets) = adjacency.get(&node) {
            for target in targets {
                if let Some(entry) = indegree.get_mut(target) {
                    *entry -= 1;
                    if *entry == 0 {
                        insert_sorted(&mut queue, target.clone());
                    }
                }
            }
        }
    }

    if order.len() != nodes.len() {
        let mut remaining: Vec<_> = nodes
            .iter()
            .filter(|layer| !order.contains(layer))
            .cloned()
            .collect();
        remaining.sort();
        let cycles = remaining.clone();
        order.extend(remaining);
        return (order, cycles);
    }

    (order, Vec::new())
}

fn layer_rank_map(order: &[String]) -> HashMap<String, usize> {
    let mut rank = HashMap::new();
    for (idx, layer) in order.iter().enumerate() {
        rank.insert(layer.clone(), idx);
    }
    rank
}

fn insert_sorted(queue: &mut VecDeque<String>, value: String) {
    let mut inserted = false;
    for idx in 0..queue.len() {
        if value < queue[idx] {
            queue.insert(idx, value.clone());
            inserted = true;
            break;
        }
    }
    if !inserted {
        queue.push_back(value);
    }
}

fn is_violation(from: &str, to: &str) -> bool {
    match (layer_prefix_value(from), layer_prefix_value(to)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

fn layer_prefix_value(layer: &str) -> Option<i32> {
    let mut chars = layer.chars();
    let mut digits = String::new();
    while let Some(ch) = chars.next() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            break;
        }
    }
    if digits.is_empty() {
        None
    } else {
        digits.parse::<i32>().ok()
    }
}

pub fn detect_layer(path: &Path) -> String {
    for component in path.components() {
        if let Some(name) = component.as_os_str().to_str() {
            if let Some(first) = name.chars().next() {
                if first.is_ascii_digit() {
                    if let Some(pos) = name.find('_') {
                        if name[..pos].chars().all(|c| c.is_ascii_digit()) {
                            return name.to_string();
                        }
                    }
                }
            }
        }
    }
    "root".to_string()
}

fn contains_tools(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str() == "tools")
}

static INCLUDE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?m)include\s*\(\s*["']([^"'\n]+)["']"#).expect("failed to compile include regex")
});
static USING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?m)(?:using|import)\s+MMSB\.([A-Za-z0-9_\.]+)"#)
        .expect("failed to compile using regex")
});
static ROOT_USING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?m)(?:using|import)\s+MMSB\s*:\s*([A-Za-z0-9_,\s]+)"#)
        .expect("failed to compile root using regex")
});
static LOCAL_USING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?m)(?:using|import)\s+\.\s*([A-Za-z0-9_]+)"#)
        .expect("failed to compile local using regex")
});

fn rust_entry_paths(root: &Path) -> BTreeSet<PathBuf> {
    ["src/lib.rs", "src/main.rs"]
        .iter()
        .map(|rel| root.join(rel))
        .filter(|p| p.exists())
        .collect()
}

fn julia_entry_paths(root: &Path) -> BTreeSet<PathBuf> {
    [
        "src/MMSB.jl",
        "src/API.jl",
        "src/MMSB/API.jl", // fallback if layout changes
    ]
    .iter()
    .map(|rel| root.join(rel))
    .filter(|p| p.exists())
    .collect()
}
