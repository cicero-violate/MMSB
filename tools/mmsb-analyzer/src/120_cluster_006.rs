//! Cluster 006: Dependency ordering and layer analysis utilities
//!
//! This module contains functions for:
//! - Julia dependency collection and file ordering
//! - Layer prefix extraction and comparison utilities
//! - Layer violation detection
//! - Directory ordering based on dependencies
//! - Cluster planning and path resolution

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::cluster_008::FunctionInfo;
#[allow(unused_imports)] pub use crate::report::collect_directory_moves;
#[allow(unused_imports)] pub use crate::report::generate_canonical_name;
// ============================================================================
// Layer Utilities (from src/010_layer_core.rs and src/020_layer_utilities.rs)
// ============================================================================

/// Extracts numeric layer prefix from a layer string (e.g., "060_file_ordering" -> 60)
pub fn layer_prefix_value(layer: &str) -> Option<i32> {
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

pub fn order_directories(
    files: &[PathBuf],
    dep_map: &HashMap<PathBuf, Vec<PathBuf>>,
) -> Vec<PathBuf> {
    let root = common_root(files);
    let mut dirs: HashSet<PathBuf> = HashSet::new();
    for file in files {
        let mut current = file.parent().map(Path::to_path_buf);
        while let Some(dir) = current {
            if let Some(ref root_path) = root {
                if !dir.starts_with(root_path) {
                    break;
                }
            }
            dirs.insert(dir.clone());
            current = dir.parent().map(Path::to_path_buf);
        }
    }

    let mut ordered: Vec<PathBuf> = dirs.into_iter().collect();
    ordered.sort_by(|a, b| crate::cluster_008::compare_path_components(a, b));

    let mut node_map = HashMap::new();
    for (idx, dir) in ordered.iter().enumerate() {
        node_map.insert(dir.clone(), idx);
    }

    let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); ordered.len()];
    let mut indegree = vec![0usize; ordered.len()];

    for (file, deps) in dep_map {
        let Some(from_dir) = file.parent().map(Path::to_path_buf) else {
            continue;
        };
        let Some(&from_idx) = node_map.get(&from_dir) else {
            continue;
        };
        for dep in deps {
            let Some(to_dir) = dep.parent().map(Path::to_path_buf) else {
                continue;
            };
            if to_dir == from_dir {
                continue;
            }
            let Some(&to_idx) = node_map.get(&to_dir) else {
                continue;
            };
            if adjacency[to_idx].insert(from_idx) {
                indegree[from_idx] += 1;
            }
        }
    }

    let mut queue: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
        .collect();

    let mut result = Vec::with_capacity(ordered.len());
    while let Some(&idx) = queue.iter().next() {
        queue.remove(&idx);
        result.push(ordered[idx].clone());
        let neighbors = adjacency[idx].clone();
        for neighbor in neighbors {
            let entry = &mut indegree[neighbor];
            if *entry > 0 {
                *entry -= 1;
                if *entry == 0 {
                    queue.insert(neighbor);
                }
            }
        }
    }

    if result.len() < ordered.len() {
        for (idx, dir) in ordered.iter().enumerate() {
            if indegree[idx] > 0 {
                result.push(dir.clone());
            }
        }
    }

    result
}

pub fn common_root(files: &[PathBuf]) -> Option<PathBuf> {
    let mut iter = files.iter();
    let first = iter.next()?.components().collect::<Vec<_>>();
    let mut prefix_len = first.len();

    for path in iter {
        let comps = path.components().collect::<Vec<_>>();
        let mut idx = 0;
        while idx < prefix_len && idx < comps.len() && comps[idx] == first[idx] {
            idx += 1;
        }
        prefix_len = idx;
    }

    if prefix_len == 0 {
        None
    } else {
        let mut root = PathBuf::new();
        for comp in first.into_iter().take(prefix_len) {
            root.push(comp.as_os_str());
        }
        Some(root)
    }
}

pub(crate) fn strip_numeric_prefix(name: &str) -> &str {
    use once_cell::sync::Lazy;
    use regex::Regex;

    static PREFIX_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^\d+_(.*)$").unwrap());
    PREFIX_RE
        .captures(name)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str())
        .unwrap_or(name)
}





// ============================================================================
// Cohesion Analysis (from src/060_cohesion_analyzer.rs)
// ============================================================================

pub fn compute_cohesion_score(
    func: &FunctionInfo,
    functions: &[FunctionInfo],
    outgoing: &HashMap<usize, usize>,
    file_layers: &HashMap<String, String>,
    call_analysis: &crate::types::CallAnalysis,
) -> f64 {
    let mut total_calls = 0usize;
    let mut intra_calls = 0usize;
    let mut external_calls = 0usize;
    let mut layer_ok = 0usize;

    for (callee_idx, count) in outgoing {
        total_calls += count;
        let callee = &functions[*callee_idx];
        if callee.file_path == func.file_path {
            intra_calls += count;
        } else {
            external_calls += count;
        }
        let current_layer = file_layers
            .get(&func.file_path)
            .cloned()
            .unwrap_or_else(|| func.layer.clone());
        let target_layer = file_layers
            .get(&callee.file_path)
            .cloned()
            .unwrap_or_else(|| callee.layer.clone());
        if crate::cluster_008::layer_adheres(&current_layer, &target_layer) {
            layer_ok += count;
        }
    }

    let total_calls_f = total_calls as f64;
    let call_locality = if total_calls == 0 {
        1.0
    } else {
        intra_calls as f64 / total_calls_f
    };
    let layer_adherence = if total_calls == 0 {
        1.0
    } else {
        layer_ok as f64 / total_calls_f
    };
    let cross_file_calls = if total_calls == 0 {
        0.0
    } else {
        external_calls as f64 / total_calls_f
    };

    let total_type_refs = call_analysis.same_file_type_refs + call_analysis.other_file_type_refs;
    let type_coupling = if total_type_refs == 0 {
        1.0
    } else {
        call_analysis.same_file_type_refs as f64 / total_type_refs as f64
    };

    let score = 0.4 * call_locality
        + 0.3 * type_coupling
        + 0.2 * layer_adherence
        - 0.1 * cross_file_calls;
    score.clamp(0.0, 1.0)
}
