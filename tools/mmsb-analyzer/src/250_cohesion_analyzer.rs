//! Function cohesion analysis and placement suggestions.

use crate::cluster_006::compute_cohesion_score;
use crate::cluster_008::{detect_layer_violation, FunctionInfo};
use crate::types::{CallAnalysis, FunctionCluster, FunctionPlacement, PlacementStatus};
use crate::types::{AnalysisResult, ElementType};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

const DEFAULT_THRESHOLD: f64 = 0.6;

#[derive(Default)]
pub struct FunctionCohesionAnalyzer {
    threshold: f64,
}

impl FunctionCohesionAnalyzer {
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_THRESHOLD,
        }
    }

    #[allow(dead_code)]
    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn analyze(&self, result: &AnalysisResult) -> Result<Vec<FunctionPlacement>> {
        let (functions, outgoing_counts, incoming_counts) = build_call_edges(result);
        let file_layers = build_function_layers(&functions);
        let (file_types, all_types) = build_type_maps(result);

        let mut placements = Vec::new();
        for (idx, func) in functions.iter().enumerate() {
            let call_analysis = build_call_analysis(
                func,
                &functions,
                &outgoing_counts[idx],
                &incoming_counts[idx],
                &file_types,
                &all_types,
            );
            let cohesion_score = compute_cohesion_score(
                func,
                &functions,
                &outgoing_counts[idx],
                &file_layers,
                &call_analysis,
            );
            let layer_violation =
                detect_layer_violation(func, &functions, &outgoing_counts[idx], &file_layers);
            let placement_status =
                determine_status(&call_analysis, cohesion_score, self.threshold, layer_violation);
            let suggested_file = match &placement_status {
                PlacementStatus::ShouldMove { .. } => suggest_file(&call_analysis),
                PlacementStatus::Orphaned { .. } => None,
                PlacementStatus::LayerViolation { .. } => None,
                PlacementStatus::Correct => None,
            };

            placements.push(FunctionPlacement {
                name: func.name.clone(),
                signature: func.signature.clone(),
                current_file: PathBuf::from(&func.file_path),
                suggested_file,
                placement_status,
                cohesion_score,
                call_analysis,
            });
        }

        Ok(placements)
    }

    pub fn detect_clusters(&self, result: &AnalysisResult) -> Result<Vec<FunctionCluster>> {
        let (functions, outgoing_counts, _) = build_call_edges(result);
        let assignments = louvain_communities(&outgoing_counts);
        let mut clusters_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, comm) in assignments.into_iter().enumerate() {
            clusters_map.entry(comm).or_default().push(idx);
        }

        let mut clusters = Vec::new();
        for component in clusters_map.values() {
            if component.len() < 3 {
                continue;
            }
            let cohesion = compute_cluster_cohesion(component, &outgoing_counts);
            let suggested_file = suggest_cluster_file(component, &functions);
            let members = component
                .iter()
                .map(|idx| {
                    format!("{}::{}", functions[*idx].file_path, functions[*idx].name)
                })
                .collect();
            clusters.push(FunctionCluster {
                members,
                cohesion,
                suggested_file,
            });
        }

        Ok(clusters)
    }
}

fn collect_functions(result: &AnalysisResult) -> Vec<FunctionInfo> {
    result
        .elements
        .iter()
        .filter(|elem| matches!(elem.element_type, ElementType::Function))
        .map(|elem| FunctionInfo {
            name: elem.name.clone(),
            signature: elem.signature.clone(),
            file_path: elem.file_path.clone(),
            layer: elem.layer.clone(),
            calls: elem.calls.clone(),
        })
        .collect()
}

fn build_call_edges(
    result: &AnalysisResult,
) -> (
    Vec<FunctionInfo>,
    Vec<HashMap<usize, usize>>,
    Vec<HashMap<usize, usize>>,
) {
    let functions = collect_functions(result);
    let name_map = build_name_map(&functions);

    let mut outgoing_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); functions.len()];
    let mut incoming_counts: Vec<HashMap<usize, usize>> = vec![HashMap::new(); functions.len()];

    for (idx, func) in functions.iter().enumerate() {
        for call in &func.calls {
            if let Some(callee_idxs) = name_map.get(call) {
                for &callee_idx in callee_idxs {
                    *outgoing_counts[idx].entry(callee_idx).or_insert(0) += 1;
                    *incoming_counts[callee_idx].entry(idx).or_insert(0) += 1;
                }
            }
        }
    }

    (functions, outgoing_counts, incoming_counts)
}

fn build_function_layers(functions: &[FunctionInfo]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for func in functions {
        map.entry(func.file_path.clone())
            .or_insert_with(|| func.layer.clone());
    }
    map
}

fn build_type_maps(
    result: &AnalysisResult,
) -> (HashMap<String, HashSet<String>>, HashSet<String>) {
    let mut file_types: HashMap<String, HashSet<String>> = HashMap::new();
    let mut all_types: HashSet<String> = HashSet::new();

    for elem in &result.elements {
        if matches!(elem.element_type, ElementType::Struct | ElementType::Enum | ElementType::Trait)
        {
            let name = elem.name.clone();
            all_types.insert(name.clone());
            file_types
                .entry(elem.file_path.clone())
                .or_default()
                .insert(name);
        }
    }

    (file_types, all_types)
}

fn build_name_map(functions: &[FunctionInfo]) -> HashMap<String, Vec<usize>> {
    let mut map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, func) in functions.iter().enumerate() {
        map.entry(func.name.clone()).or_default().push(idx);
    }
    map
}

fn build_call_analysis(
    func: &FunctionInfo,
    functions: &[FunctionInfo],
    outgoing: &HashMap<usize, usize>,
    incoming: &HashMap<usize, usize>,
    file_types: &HashMap<String, HashSet<String>>,
    all_types: &HashSet<String>,
) -> CallAnalysis {
    let mut intra_file_calls = 0usize;
    let mut inter_file_calls: BTreeMap<PathBuf, usize> = BTreeMap::new();

    for (callee_idx, count) in outgoing {
        let callee = &functions[*callee_idx];
        if callee.file_path == func.file_path {
            intra_file_calls += count;
        } else {
            *inter_file_calls
                .entry(PathBuf::from(&callee.file_path))
                .or_insert(0) += count;
        }
    }

    let mut calls_from_same_file = 0usize;
    let mut calls_from_other_files: BTreeMap<PathBuf, usize> = BTreeMap::new();
    for (caller_idx, count) in incoming {
        let caller = &functions[*caller_idx];
        if caller.file_path == func.file_path {
            calls_from_same_file += count;
        } else {
            *calls_from_other_files
                .entry(PathBuf::from(&caller.file_path))
                .or_insert(0) += count;
        }
    }

    let (same_file_type_refs, other_file_type_refs) =
        compute_type_coupling(func, file_types, all_types);

    CallAnalysis {
        intra_file_calls,
        inter_file_calls: inter_file_calls.into_iter().collect(),
        calls_from_same_file,
        calls_from_other_files: calls_from_other_files.into_iter().collect(),
        same_file_type_refs,
        other_file_type_refs,
    }
}

fn determine_status(
    call_analysis: &CallAnalysis,
    cohesion_score: f64,
    threshold: f64,
    layer_violation: Option<(String, String)>,
) -> PlacementStatus {
    let total_activity =
        call_analysis.intra_file_calls + call_analysis.calls_from_same_file;
    let external_activity: usize = call_analysis
        .calls_from_other_files
        .iter()
        .map(|(_, count)| *count)
        .sum();

    if total_activity == 0 && external_activity == 0 {
        return PlacementStatus::Orphaned {
            suggested_module: "utilities".to_string(),
        };
    }

    if let Some((current_layer, required_layer)) = layer_violation {
        return PlacementStatus::LayerViolation {
            current_layer,
            required_layer,
        };
    }

    if cohesion_score >= threshold {
        return PlacementStatus::Correct;
    }

    let mut impact = threshold - cohesion_score;
    if impact < 0.0 {
        impact = 0.0;
    }

    PlacementStatus::ShouldMove {
        reason: format!(
            "cohesion {:.2} below threshold {:.2}",
            cohesion_score, threshold
        ),
        impact,
    }
}

fn suggest_file(call_analysis: &CallAnalysis) -> Option<PathBuf> {
    let mut best_file: Option<PathBuf> = None;
    let mut best_score = 0usize;

    for (file, count) in &call_analysis.calls_from_other_files {
        if *count > best_score {
            best_score = *count;
            best_file = Some(file.clone());
        }
    }

    if best_score == 0 {
        return None;
    }

    if let Some(candidate) = &best_file {
        let outgoing_to_candidate = call_analysis
            .inter_file_calls
            .iter()
            .find(|(path, _)| path == candidate)
            .map(|(_, count)| *count)
            .unwrap_or(0);
        if outgoing_to_candidate == 0 && call_analysis.intra_file_calls >= best_score {
            return None;
        }
    }

    best_file
}

fn compute_cluster_cohesion(
    members: &[usize],
    outgoing_counts: &[HashMap<usize, usize>],
) -> f64 {
    let member_set: HashSet<usize> = members.iter().copied().collect();
    let mut internal = 0usize;
    let mut total = 0usize;

    for idx in members {
        if let Some(outgoing) = outgoing_counts.get(*idx) {
            for (target, count) in outgoing {
                total += count;
                if member_set.contains(target) {
                    internal += count;
                }
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        internal as f64 / total as f64
    }
}

fn suggest_cluster_file(
    members: &[usize],
    functions: &[FunctionInfo],
) -> Option<PathBuf> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for idx in members {
        let path = functions[*idx].file_path.as_str();
        *counts.entry(path).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(path, _)| PathBuf::from(path))
}

fn compute_type_coupling(
    func: &FunctionInfo,
    file_types: &HashMap<String, HashSet<String>>,
    all_types: &HashSet<String>,
) -> (usize, usize) {
    let mut same_file = 0usize;
    let mut other_file = 0usize;
    let tokens = extract_identifiers(&func.signature);
    let same_set = file_types.get(&func.file_path);

    for token in tokens {
        if let Some(types) = same_set {
            if types.contains(&token) {
                same_file += 1;
                continue;
            }
        }
        if all_types.contains(&token) {
            other_file += 1;
        }
    }

    (same_file, other_file)
}

fn extract_identifiers(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn louvain_communities(outgoing_counts: &[HashMap<usize, usize>]) -> Vec<usize> {
    let n = outgoing_counts.len();
    if n == 0 {
        return Vec::new();
    }

    let (neighbors, degrees, total_weight) = build_undirected_graph(outgoing_counts);
    if total_weight == 0 {
        return (0..n).collect();
    }

    let two_m = (2 * total_weight) as f64;
    let mut community: Vec<usize> = (0..n).collect();
    let mut sum_tot = degrees.clone();

    let max_iters = 25;
    for iter in 0..max_iters {
        let mut moved = false;
        for node in 0..n {
            let current = community[node];
            let mut neighbor_comms: HashMap<usize, usize> = HashMap::new();
            for (neighbor, weight) in &neighbors[node] {
                let comm = community[*neighbor];
                *neighbor_comms.entry(comm).or_insert(0) += *weight;
            }

            let mut best_comm = current;
            let mut best_gain = 0.0f64;
            for (comm, k_i_in) in neighbor_comms {
                if comm == current {
                    continue;
                }
                let gain = (k_i_in as f64
                    - (degrees[node] as f64 * sum_tot[comm] as f64) / two_m)
                    / two_m;
                if gain > best_gain {
                    best_gain = gain;
                    best_comm = comm;
                }
            }

            if best_comm != current && best_gain > 0.0 {
                community[node] = best_comm;
                sum_tot[current] = sum_tot[current].saturating_sub(degrees[node]);
                sum_tot[best_comm] += degrees[node];
                moved = true;
            }
        }
        if !moved {
            break;
        }
        if iter % 5 == 0 {
            println!("  Louvain pass {}...", iter + 1);
        }
    }

    community
}

fn build_undirected_graph(
    outgoing_counts: &[HashMap<usize, usize>],
) -> (Vec<Vec<(usize, usize)>>, Vec<usize>, usize) {
    let n = outgoing_counts.len();
    let mut edge_weights: HashMap<(usize, usize), usize> = HashMap::new();

    for (src, outgoing) in outgoing_counts.iter().enumerate() {
        for (dst, weight) in outgoing {
            let (a, b) = if src <= *dst { (src, *dst) } else { (*dst, src) };
            *edge_weights.entry((a, b)).or_insert(0) += *weight;
        }
    }

    let mut neighbors = vec![Vec::new(); n];
    let mut degrees = vec![0usize; n];
    let mut total_weight = 0usize;

    for ((a, b), weight) in edge_weights {
        if a == b {
            continue;
        }
        neighbors[a].push((b, weight));
        neighbors[b].push((a, weight));
        degrees[a] += weight;
        degrees[b] += weight;
        total_weight += weight;
    }

    (neighbors, degrees, total_weight)
}
