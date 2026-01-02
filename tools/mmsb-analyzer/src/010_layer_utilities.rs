//! Cluster 008: Core dependency analysis and layer utilities
//!
//! This module contains foundational functions for:
//! - Rust file dependency ordering and resolution
//! - Layer violation detection
//! - Layer name extraction
//! - Graph visualization node styling
//! - Cyclomatic complexity calculation
//!
//! Functions moved from multiple modules as part of Phase 2 refactoring.

use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use petgraph::graph::DiGraph;
use petgraph::visit::EdgeRef;

use crate::dependency::{
    LayerEdge, LayerGraph, ReferenceDetail, UnresolvedDependency,
};
use crate::types::{FileLayerViolation, NodeType};

// ============================================================================
// From src/000_dependency.rs
// ============================================================================

pub fn build_result(
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
        let mmsb_a = is_mmsb_main(a);
        let mmsb_b = is_mmsb_main(b);
        if mmsb_a && !mmsb_b {
            return Ordering::Less;
        } else if mmsb_b && !mmsb_a {
            return Ordering::Greater;
        }
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
            violation: is_layer_violation(&from, &to),
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

fn is_mmsb_main(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n == "MMSB.jl")
        .unwrap_or(false)
}

// ============================================================================
// From src/010_layer_core.rs
// ============================================================================

/// Checks if a dependency from one layer to another violates layer ordering
/// Returns true if from_layer > to_layer (violation: higher depends on lower)
pub fn is_layer_violation(from: &str, to: &str) -> bool {
    match (layer_prefix_value(from), layer_prefix_value(to)) {
        (Some(a), Some(b)) => a > b,
        _ => false,
    }
}

/// Extracts numeric layer prefix from a layer string (e.g., "060_file_ordering" -> 60)
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

// ============================================================================
// From src/010_layer_core.rs (continued)
// ============================================================================

pub fn compare_dir_layers(a: &Path, b: &Path) -> Ordering {
    let a_name = a.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let b_name = b.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let a_layer = layer_prefix_value(a_name).unwrap_or(i32::MAX);
    let b_layer = layer_prefix_value(b_name).unwrap_or(i32::MAX);
    a_layer.cmp(&b_layer).then_with(|| a_name.cmp(b_name))
}

pub fn compare_path_components(a: &Path, b: &Path) -> Ordering {
    let a_components: Vec<_> = a.components().collect();
    let b_components: Vec<_> = b.components().collect();
    let min_len = a_components.len().min(b_components.len());

    for idx in 0..min_len {
        let a_name = a_components[idx].as_os_str().to_string_lossy();
        let b_name = b_components[idx].as_os_str().to_string_lossy();
        let a_prefix = layer_prefix_value(&a_name);
        let b_prefix = layer_prefix_value(&b_name);
        let cmp = match (a_prefix, b_prefix) {
            (Some(a_val), Some(b_val)) => a_val.cmp(&b_val),
            _ => a_name.cmp(&b_name),
        };
        if cmp != Ordering::Equal {
            return cmp;
        }
    }

    a_components.len().cmp(&b_components.len())
}

pub fn layer_adheres(current_layer: &str, target_layer: &str) -> bool {
    match (layer_prefix_value(current_layer), layer_prefix_value(target_layer)) {
        (Some(curr), Some(target)) => curr <= target,
        _ => true,
    }
}

pub(crate) fn structural_layer_value(layer: &Option<String>, default: i32) -> i32 {
    layer
        .as_ref()
        .and_then(|value| layer_prefix_value(value))
        .unwrap_or(default)
}

pub fn detect_layer_violations(
    graph: &DiGraph<PathBuf, ()>,
    file_layers: &HashMap<PathBuf, String>,
) -> Vec<FileLayerViolation> {
    let mut violations = Vec::new();
    for edge in graph.edge_references() {
        let from = &graph[edge.source()];
        let to = &graph[edge.target()];
        let from_layer = file_layers
            .get(from)
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        let to_layer = file_layers
            .get(to)
            .cloned()
            .unwrap_or_else(|| "root".to_string());
        if let (Some(from_val), Some(to_val)) =
            (layer_prefix_value(&from_layer), layer_prefix_value(&to_layer))
        {
            if from_val > to_val {
                violations.push(FileLayerViolation {
                    from: from.clone(),
                    to: to.clone(),
                    from_layer,
                    to_layer,
                });
            }
        }
    }
    violations
}

#[derive(Clone)]
pub struct FunctionInfo {
    pub name: String,
    pub signature: String,
    pub file_path: String,
    pub layer: String,
    pub calls: Vec<String>,
}

pub fn detect_layer_violation(
    func: &FunctionInfo,
    functions: &[FunctionInfo],
    outgoing: &HashMap<usize, usize>,
    file_layers: &HashMap<String, String>,
) -> Option<(String, String)> {
    let current_layer = file_layers
        .get(&func.file_path)
        .cloned()
        .unwrap_or_else(|| func.layer.clone());
    let current_value = layer_prefix_value(&current_layer)?;

    let mut violation: Option<(i32, String)> = None;
    for (callee_idx, _) in outgoing {
        let callee = &functions[*callee_idx];
        let target_layer = file_layers
            .get(&callee.file_path)
            .cloned()
            .unwrap_or_else(|| callee.layer.clone());
        if let Some(target_value) = layer_prefix_value(&target_layer) {
            if target_value < current_value {
                match violation {
                    Some((best_value, _)) if target_value >= best_value => {}
                    _ => {
                        violation = Some((target_value, target_layer));
                    }
                }
            }
        }
    }

    violation.map(|(_, target_layer)| (current_layer, target_layer))
}

// ============================================================================
// From src/020_layer_utilities.rs (report planning helpers)
// ============================================================================

pub fn parse_cluster_members(
    cluster: &crate::types::FunctionCluster,
) -> Vec<crate::report::ClusterMember> {
    cluster
        .members
        .iter()
        .filter_map(|member| {
            let (file, name) = member.rsplit_once("::")?;
            Some(crate::report::ClusterMember {
                file: PathBuf::from(file),
                name: name.to_string(),
            })
        })
        .collect()
}

pub fn is_core_module_path(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|name| name.to_str()) else {
        return false;
    };
    stem.starts_with("040_dependency") || stem.starts_with("060_layer_core")
}

pub fn cluster_target_path(
    target: PathBuf,
    members: &[crate::report::ClusterMember],
    root_path: &Path,
    idx: usize,
) -> PathBuf {
    if !is_core_module_path(&target) {
        return target;
    }
    let prefix = target
        .file_stem()
        .and_then(|name| name.to_str())
        .and_then(|stem| layer_prefix_value(stem))
        .unwrap_or(900);
    let file_name = format!("{:03}_cluster_{:03}.rs", prefix, idx + 1);
    let dir = members
        .first()
        .and_then(|member| member.file.parent())
        .unwrap_or(root_path);
    dir.join(file_name)
}

pub fn collect_cluster_plans(
    clusters: &[crate::types::FunctionCluster],
    root_path: &Path,
) -> Vec<crate::report::ClusterPlan> {
    let mut plans = Vec::new();
    for (idx, cluster) in clusters.iter().enumerate() {
        let all_members = parse_cluster_members(cluster);
        let target = if let Some(suggested) = &cluster.suggested_file {
            suggested.clone()
        } else if let Some(first) = all_members.first() {
            let file_name = format!("900_cluster_{:03}.rs", idx + 1);
            first
                .file
                .parent()
                .unwrap_or(root_path)
                .join(file_name)
        } else {
            let file_name = format!("900_cluster_{:03}.rs", idx + 1);
            root_path.join(file_name)
        };
        let target = cluster_target_path(target, &all_members, root_path, idx);
        let members = all_members
            .into_iter()
            .filter(|member| member.file != target)
            .collect::<Vec<_>>();
        if members.len() < 2 {
            continue;
        }
        plans.push(crate::report::ClusterPlan {
            target,
            cohesion: cluster.cohesion,
            members,
        });
    }
    plans.sort_by(|a, b| {
        use std::cmp::Ordering;
        b.cohesion
            .partial_cmp(&a.cohesion)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.members.len().cmp(&a.members.len()))
            .then_with(|| a.target.cmp(&b.target))
    });
    plans
}

// ============================================================================
// From src/120_dot_exporter.rs
// ============================================================================

pub fn node_style(node_type: &NodeType) -> (&str, &str, &str) {
    match node_type {
        NodeType::Entry => ("ellipse", "lightgreen", "\"filled,bold\""),
        NodeType::Exit => ("doubleoctagon", "lightcoral", "\"filled,bold\""),
        NodeType::BasicBlock => ("box", "lightblue", "filled"),
        NodeType::Branch => ("diamond", "yellow", "filled"),
        NodeType::LoopHeader => ("box", "orange", "\"filled,rounded\""),
    }
}

pub fn cyclomatic_complexity(cfg: &crate::types::FunctionCfg) -> usize {
    let edges = cfg.edges.len() as isize;
    let nodes = cfg.nodes.len() as isize;
    let exits = 1isize; // assume one exit
    let cc = edges - nodes + 2 * exits;
    if cc <= 0 {
        1
    } else {
        cc as usize
    }
}

pub fn structural_cmp(a: &crate::report::PlanItem, b: &crate::report::PlanItem) -> std::cmp::Ordering {
    let a_required = structural_layer_value(&a.required_layer, i32::MAX);
    let b_required = structural_layer_value(&b.required_layer, i32::MAX);
    let a_current = structural_layer_value(&a.current_layer, i32::MIN);
    let b_current = structural_layer_value(&b.current_layer, i32::MIN);
    let a_benefit = if a.cost == 0 {
        0
    } else {
        (a.benefit.saturating_mul(1000)) / a.cost
    };
    let b_benefit = if b.cost == 0 {
        0
    } else {
        (b.benefit.saturating_mul(1000)) / b.cost
    };
    a_required
        .cmp(&b_required)
        .then_with(|| b.is_utility.cmp(&a.is_utility))
        .then_with(|| b_benefit.cmp(&a_benefit))
        .then_with(|| b.impact_weight.cmp(&a.impact_weight))
        .then_with(|| b_current.cmp(&a_current))
        .then_with(|| a.description.cmp(&b.description))
}

pub fn sort_structural_items(items: &mut Vec<crate::report::PlanItem>) {
    use std::collections::HashMap;
    use std::path::PathBuf;

    if items.len() <= 1 {
        return;
    }

    let count = items.len();
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut indegree = vec![0usize; count];

    let mut file_to_items: HashMap<PathBuf, Vec<usize>> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        if let Some(path) = &item.current_file {
            file_to_items.entry(path.clone()).or_default().push(idx);
        }
    }

    for i in 0..count {
        for j in (i + 1)..count {
            let req_i = structural_layer_value(&items[i].required_layer, i32::MAX);
            let req_j = structural_layer_value(&items[j].required_layer, i32::MAX);
            let mut edge = None;
            if req_i != req_j {
                edge = if req_i < req_j { Some((i, j)) } else { Some((j, i)) };
            } else if items[i].is_utility != items[j].is_utility {
                edge = if items[i].is_utility {
                    Some((i, j))
                } else {
                    Some((j, i))
                };
            }
            if let Some((from, to)) = edge {
                edges[from].push(to);
                indegree[to] += 1;
            }
        }
    }

    for (idx, item) in items.iter().enumerate() {
        for file in &item.outgoing_files {
            if let Some(dependents) = file_to_items.get(file) {
                for &dependent_idx in dependents {
                    if dependent_idx == idx {
                        continue;
                    }
                    edges[dependent_idx].push(idx);
                    indegree[idx] += 1;
                }
            }
        }
    }

    let mut ordered_indices = Vec::with_capacity(count);
    let mut available: Vec<usize> = (0..count).filter(|&i| indegree[i] == 0).collect();
    while !available.is_empty() {
        available.sort_by(|&a, &b| structural_cmp(&items[a], &items[b]));
        let next = available.remove(0);
        ordered_indices.push(next);
        for &neighbor in &edges[next] {
            indegree[neighbor] = indegree[neighbor].saturating_sub(1);
            if indegree[neighbor] == 0 {
                available.push(neighbor);
            }
        }
    }

    if ordered_indices.len() != count {
        items.sort_by(structural_cmp);
        return;
    }

    let mut reordered = Vec::with_capacity(count);
    for idx in ordered_indices {
        reordered.push(items[idx].clone());
    }
    *items = reordered;
}
