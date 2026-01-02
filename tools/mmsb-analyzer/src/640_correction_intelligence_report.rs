#![allow(dead_code)]
//! Correction intelligence report generator.

use crate::correction_plan_types::{
    CorrectionPlan,
    CorrectionStrategy,
    ErrorTier,
    RefactorAction,
    Severity,
    ViolationPrediction,
    ViolationType,
};
use crate::quality_delta_calculator::Metrics;
use crate::quality_delta_types::{QualityDelta, RollbackCriteria};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::{collections::HashSet, fs};

use crate::invariant_types::InvariantAnalysisResult;
use crate::types::{AnalysisResult, CallGraphNode, CodeElement};
use crate::action_impact_estimator::AnalysisState as ImpactState;
#[allow(unused_imports)] pub use crate::correction_plan_serializer::write_intelligence_outputs;

#[allow(unused_imports)] pub use crate::correction_plan_serializer::write_intelligence_outputs_at;
use crate::action_impact_estimator::AnalysisState;
use crate::correction_plan_types::VisibilityPlanOption;
use crate::tier_classifier::classify_tier;
use crate::verification_policy_types::VerificationCheck;
use crate::verification_policy_types::VerificationPolicy;
use crate::verification_policy_types::VerificationScope;
use crate::quality_delta_types::RollbackCondition;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionIntelligenceReport {
    pub version: String,
    pub timestamp: String,
    pub project_root: PathBuf,
    pub actions_analyzed: usize,
    pub correction_plans: Vec<CorrectionPlan>,
    pub verification_policies: Vec<crate::verification_policy_types::VerificationPolicy>,
    pub rollback_criteria: Vec<RollbackCriteria>,
    pub quality_deltas: Vec<QualityDelta>,
    pub summary: CorrectionSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrectionSummary {
    pub trivial_count: usize,
    pub moderate_count: usize,
    pub complex_count: usize,
    pub total_predicted_violations: usize,
    pub average_confidence: f64,
    pub estimated_total_fix_time_seconds: u32,
}

#[derive(Clone, Debug)]
pub struct IntelligenceState<'a> {
    pub root: PathBuf,
    pub invariants: &'a InvariantAnalysisResult,
    pub call_graph: &'a HashMap<String, CallGraphNode>,
    pub elements: &'a [CodeElement],
    pub metrics: Metrics,
}

pub fn build_state<'a>(
    root: &'a Path,
    analysis: &'a AnalysisResult,
    metrics: Metrics,
) -> IntelligenceState<'a> {
    IntelligenceState {
        root: root.to_path_buf(),
        invariants: &analysis.invariants,
        call_graph: &analysis.call_graph,
        elements: &analysis.elements,
        metrics,
    }
}





pub fn filter_path_coherence_report(
    report: &CorrectionIntelligenceReport,
) -> CorrectionIntelligenceReport {
    let mut plans = Vec::new();
    let mut policies = Vec::new();
    let mut criteria = Vec::new();
    let mut deltas = Vec::new();

    for (idx, plan) in report.correction_plans.iter().enumerate() {
        let mut has_path_coherence = false;
        for strategy in &plan.strategies {
            match strategy {
                CorrectionStrategy::UpdatePath { .. } => {
                    has_path_coherence = true;
                    break;
                }
                CorrectionStrategy::UpdateCaller { old_ref, .. } => {
                    let trimmed = old_ref.trim_start();
                    if trimmed.starts_with("mod ")
                        || trimmed.starts_with("pub mod ")
                        || trimmed.starts_with("use ")
                        || trimmed.starts_with("#[path")
                    {
                        has_path_coherence = true;
                        break;
                    }
                }
                _ => {}
            }
        }
        let is_rename_file = plan.action_id.starts_with("rename_file_");

        if !(has_path_coherence || is_rename_file) {
            continue;
        }

        plans.push(plan.clone());
        if let Some(policy) = report.verification_policies.get(idx) {
            policies.push(policy.clone());
        }
        if let Some(rollback) = report.rollback_criteria.get(idx) {
            criteria.push(rollback.clone());
        }
        if let Some(delta) = report.quality_deltas.get(idx) {
            deltas.push(delta.clone());
        }
    }

    let summary = compute_summary(&plans, &deltas);

    CorrectionIntelligenceReport {
        version: report.version.clone(),
        timestamp: report.timestamp.clone(),
        project_root: report.project_root.clone(),
        actions_analyzed: plans.len(),
        correction_plans: plans,
        verification_policies: policies,
        rollback_criteria: criteria,
        quality_deltas: deltas,
        summary,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdmissionPreflightReport {
    pub version: String,
    pub timestamp: String,
    pub project_root: PathBuf,
    pub entries: Vec<AdmissionPreflightEntry>,
    pub summary: AdmissionPreflightSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdmissionPreflightEntry {
    pub action_id: String,
    pub function: String,
    pub target_layer: String,
    pub admissible: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdmissionPreflightSummary {
    pub total_moves: usize,
    pub admissible: usize,
    pub blocked: usize,
}

pub fn write_admission_preflight_report(
    report: &CorrectionIntelligenceReport,
    root: &Path,
    output_dir: &Path,
) -> std::io::Result<()> {
    let preflight = generate_admission_preflight(report, root);
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join("admission_preflight.json");
    std::fs::write(path, serde_json::to_string_pretty(&preflight)?)?;
    Ok(())
}

fn generate_admission_preflight(
    report: &CorrectionIntelligenceReport,
    root: &Path,
) -> AdmissionPreflightReport {
    let mut entries = Vec::new();
    for plan in &report.correction_plans {
        for strategy in &plan.strategies {
            if let CorrectionStrategy::MoveToLayer {
                function,
                target_layer,
            } = strategy
            {
                let decision = evaluate_move_admission(root, function, &plan.action_id);
                entries.push(AdmissionPreflightEntry {
                    action_id: plan.action_id.clone(),
                    function: function.clone(),
                    target_layer: target_layer.clone(),
                    admissible: decision.admissible,
                    reasons: decision.reasons,
                });
            }
        }
    }
    let admissible = entries.iter().filter(|entry| entry.admissible).count();
    let summary = AdmissionPreflightSummary {
        total_moves: entries.len(),
        admissible,
        blocked: entries.len().saturating_sub(admissible),
    };
    AdmissionPreflightReport {
        version: report.version.clone(),
        timestamp: report.timestamp.clone(),
        project_root: report.project_root.clone(),
        entries,
        summary,
    }
}

struct AdmissionDecision {
    admissible: bool,
    reasons: Vec<String>,
}

fn evaluate_move_admission(root: &Path, function: &str, action_id: &str) -> AdmissionDecision {
    let mut reasons = Vec::new();
    let candidates = match find_function_definition_candidates(root, function) {
        Ok(paths) => paths,
        Err(err) => {
            reasons.push(err);
            return AdmissionDecision {
                admissible: false,
                reasons,
            };
        }
    };
    if candidates.len() > 1 {
        let locations = candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        reasons.push(format!(
            "MoveToLayer blocked for {}: ambiguous definitions [{}]",
            function, locations
        ));
        return AdmissionDecision {
            admissible: false,
            reasons,
        };
    }
    let source_path = match candidates.into_iter().next() {
        Some(path) => path,
        None => {
            reasons.push(format!("MoveToLayer: function {} not found", function));
            return AdmissionDecision {
                admissible: false,
                reasons,
            };
        }
    };
    let contents = match fs::read_to_string(&source_path) {
        Ok(text) => text,
        Err(err) => {
            reasons.push(format!("Failed to read {:?}: {}", source_path, err));
            return AdmissionDecision {
                admissible: false,
                reasons,
            };
        }
    };
    if function_in_impl_block(&contents, function) {
        reasons.push(format!(
            "MoveToLayer blocked for {}: function inside impl/trait block",
            function
        ));
        return AdmissionDecision {
            admissible: false,
            reasons,
        };
    }
    match is_test_scoped_function(&contents, function) {
        Ok(true) => {
            reasons.push(format!(
                "MoveToLayer blocked for {}: test-scoped function",
                function
            ));
            return AdmissionDecision {
                admissible: false,
                reasons,
            };
        }
        Err(err) => {
            reasons.push(err);
            return AdmissionDecision {
                admissible: false,
                reasons,
            };
        }
        Ok(false) => {}
    }

    match find_private_dependencies(&contents, function, &source_path) {
        Ok(blockers) if !blockers.is_empty() => {
            let blocker_list = blockers
                .into_iter()
                .map(|b| format!("{} ({}; {}) @ {}", b.name, b.kind, b.visibility, b.path))
                .collect::<Vec<_>>()
                .join("; ");
            reasons.push(format!(
                "MoveToLayer blocked for {}: RequiresEnsureDependencies [{}]",
                function, blocker_list
            ));
            reasons.push(format!(
                "Admission blocked by C12 pre-flight guard: {}",
                action_id
            ));
            AdmissionDecision {
                admissible: false,
                reasons,
            }
        }
        Ok(_) => AdmissionDecision {
            admissible: true,
            reasons,
        },
        Err(err) => {
            reasons.push(err);
            AdmissionDecision {
                admissible: false,
                reasons,
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DependencyBlocker {
    name: String,
    kind: String,
    visibility: String,
    path: String,
}

fn find_private_dependencies(
    contents: &str,
    function: &str,
    source_path: &Path,
) -> Result<Vec<DependencyBlocker>, String> {
    let function_block = extract_function_block_from_contents(contents, function)?;
    let identifiers = extract_identifiers(&function_block);
    let mut blockers = Vec::new();
    let mut seen = HashSet::new();

    for line in contents.lines() {
        let trimmed = line.trim_start();
        let (visibility, rest) = if trimmed.starts_with("pub(") {
            ("pub(...)".to_string(), trimmed)
        } else if let Some(after) = trimmed.strip_prefix("pub ") {
            ("pub".to_string(), after)
        } else {
            ("private".to_string(), trimmed)
        };
        let (kind, name) = if let Some(rest) = rest.strip_prefix("fn ") {
            ("fn", rest)
        } else if let Some(rest) = rest.strip_prefix("struct ") {
            ("struct", rest)
        } else if let Some(rest) = rest.strip_prefix("enum ") {
            ("enum", rest)
        } else if let Some(rest) = rest.strip_prefix("type ") {
            ("type", rest)
        } else if let Some(rest) = rest.strip_prefix("const ") {
            ("const", rest)
        } else if let Some(rest) = rest.strip_prefix("static ") {
            ("static", rest)
        } else {
            continue;
        };

        let symbol = name
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or("");
        if symbol.is_empty() || symbol == function {
            continue;
        }
        if identifiers.contains(symbol) && !visibility.starts_with("pub") {
            if seen.insert(symbol.to_string()) {
                blockers.push(DependencyBlocker {
                    name: symbol.to_string(),
                    kind: kind.to_string(),
                    visibility,
                    path: source_path.display().to_string(),
                });
            }
        }
    }

    Ok(blockers)
}

fn extract_function_block_from_contents(contents: &str, function: &str) -> Result<String, String> {
    let mut start_idx = None;
    let mut offset = 0usize;
    for line in contents.lines() {
        let line_len = line.len() + 1;
        if is_function_signature_line(line, function) {
            start_idx = Some(offset);
            break;
        }
        offset += line_len;
    }
    let start = start_idx.ok_or_else(|| {
        format!(
            "MoveToLayer: function {} signature not found in {}",
            function, "source"
        )
    })?;

    let bytes = contents.as_bytes();
    let mut brace_start = None;
    for idx in start..bytes.len() {
        if bytes[idx] == b'{' {
            brace_start = Some(idx);
            break;
        }
    }
    let brace_start = brace_start.ok_or_else(|| {
        format!("MoveToLayer: no body found for {} in source", function)
    })?;

    let mut depth = 0i32;
    let mut end_idx = None;
    for idx in brace_start..bytes.len() {
        match bytes[idx] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(idx + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end_idx.ok_or_else(|| {
        format!("MoveToLayer: unterminated body for {} in source", function)
    })?;
    Ok(contents[start..end].to_string())
}

fn extract_identifiers(contents: &str) -> HashSet<String> {
    let mut idents = HashSet::new();
    let mut current = String::new();
    for ch in contents.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                if is_identifier_candidate(&current) {
                    idents.insert(current.clone());
                }
                current.clear();
            }
        }
    }
    if !current.is_empty() && is_identifier_candidate(&current) {
        idents.insert(current);
    }
    idents
}

fn is_identifier_candidate(value: &str) -> bool {
    if value.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
        return false;
    }
    let lower = value.to_lowercase();
    let keywords = [
        "fn", "let", "mut", "pub", "use", "crate", "self", "super", "mod", "impl", "trait",
        "where", "for", "while", "loop", "if", "else", "match", "return", "break", "continue",
        "struct", "enum", "type", "const", "static", "move", "async", "await", "dyn", "ref",
        "in", "as",
    ];
    if keywords.contains(&lower.as_str()) {
        return false;
    }
    let primitives = [
        "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char", "str", "string", "vec", "option", "result",
    ];
    !primitives.contains(&lower.as_str())
}

fn find_function_definition_candidates(root: &Path, function: &str) -> Result<Vec<PathBuf>, String> {
    let search_root = if root.join("src").is_dir() {
        root.join("src")
    } else {
        root.to_path_buf()
    };
    let mut matches = Vec::new();
    let mut stack = vec![search_root];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|e| format!("Failed to read {:?}: {}", dir, e))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read dir entry: {}", e))?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name == "target" || name == ".git" || name == "_old" {
                        continue;
                    }
                }
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let contents = fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read {:?}: {}", path, e))?;
            if function_signature_found(&contents, function) {
                if function_in_impl_block(&contents, function) {
                    return Err(format!(
                        "MoveToLayer: function {} is inside an impl/trait block; refused",
                        function
                    ));
                }
                matches.push(path);
            }
        }
    }
    if matches.is_empty() {
        return Err(format!("MoveToLayer: function {} not found", function));
    }
    Ok(matches)
}

fn function_signature_found(contents: &str, function: &str) -> bool {
    contents
        .lines()
        .any(|line| is_function_signature_line(line, function))
}

fn is_function_signature_line(line: &str, function: &str) -> bool {
    let trimmed = line.trim_start();
    if !trimmed.contains("fn ") {
        return false;
    }
    let patterns = [
        format!("fn {}(", function),
        format!("fn {}<", function),
        format!("fn {} ", function),
    ];
    patterns.iter().any(|pattern| trimmed.contains(pattern))
}

fn function_in_impl_block(contents: &str, function: &str) -> bool {
    let mut impl_depth = 0i32;
    for line in contents.lines() {
        if is_function_signature_line(line, function) {
            return impl_depth > 0;
        }
        let trimmed = line.trim_start();
        if (trimmed.starts_with("impl ") || trimmed.starts_with("impl<") || trimmed.starts_with("trait "))
            && trimmed.contains('{')
        {
            impl_depth += 1;
        }
        if trimmed.contains('}') && impl_depth > 0 {
            impl_depth -= 1;
        }
    }
    false
}

fn is_test_attribute_line(trimmed: &str) -> bool {
    if !trimmed.starts_with("#[") {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    lower.contains("test") && !lower.contains("cfg")
}

fn is_test_scoped_function(contents: &str, function: &str) -> Result<bool, String> {
    let mut brace_depth: i32 = 0;
    let mut attr_test = false;
    let mut pending_cfg_test = false;
    let mut pending_test_mod = false;
    let mut test_mod_depths: Vec<i32> = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            pending_cfg_test = true;
        }
        if pending_cfg_test && trimmed.starts_with("mod ") {
            pending_test_mod = true;
            pending_cfg_test = false;
        }
        if pending_test_mod && trimmed.contains('{') {
            test_mod_depths.push(brace_depth + 1);
            pending_test_mod = false;
        }
        if is_test_attribute_line(trimmed) {
            attr_test = true;
        }

        if is_function_signature_line(line, function) {
            if attr_test || !test_mod_depths.is_empty() {
                return Ok(true);
            }
            return Ok(false);
        }

        if !trimmed.starts_with("#[") && !trimmed.is_empty() && !trimmed.starts_with("//") {
            attr_test = false;
        }

        let opens = line.chars().filter(|c| *c == '{').count() as i32;
        let closes = line.chars().filter(|c| *c == '}').count() as i32;
        brace_depth += opens - closes;

        while let Some(&depth) = test_mod_depths.last() {
            if brace_depth < depth {
                test_mod_depths.pop();
            } else {
                break;
            }
        }
    }

    Ok(false)
}

pub fn filter_visibility_report(
    report: &CorrectionIntelligenceReport,
) -> CorrectionIntelligenceReport {
    let mut plans = Vec::new();
    let mut policies = Vec::new();
    let mut criteria = Vec::new();
    let mut deltas = Vec::new();

    for (idx, plan) in report.correction_plans.iter().enumerate() {
        let mut has_visibility = false;
        for strategy in &plan.strategies {
            match strategy {
                CorrectionStrategy::AdjustVisibility { .. } => {
                    has_visibility = true;
                    break;
                }
                CorrectionStrategy::VisibilityPlan { .. } => {
                    has_visibility = true;
                    break;
                }
                CorrectionStrategy::ManualReview { reason, .. }
                    if reason.starts_with("review:") =>
                {
                    has_visibility = true;
                    break;
                }
                _ => {}
            }
        }
        if !has_visibility {
            continue;
        }
        plans.push(plan.clone());
        if let Some(policy) = report.verification_policies.get(idx) {
            policies.push(policy.clone());
        }
        if let Some(rollback) = report.rollback_criteria.get(idx) {
            criteria.push(rollback.clone());
        }
        if let Some(delta) = report.quality_deltas.get(idx) {
            deltas.push(delta.clone());
        }
    }

    let summary = compute_summary(&plans, &deltas);

    CorrectionIntelligenceReport {
        version: report.version.clone(),
        timestamp: report.timestamp.clone(),
        project_root: report.project_root.clone(),
        actions_analyzed: plans.len(),
        correction_plans: plans,
        verification_policies: policies,
        rollback_criteria: criteria,
        quality_deltas: deltas,
        summary,
    }
}

#[derive(Clone, Debug)]
struct ClusterMove {
    function: String,
    from: PathBuf,
}

#[derive(Clone, Debug)]
struct ClusterBatch {
    target: PathBuf,
    moves: Vec<ClusterMove>,
}

fn parse_phase2_cluster_plan(plan_path: &Path) -> std::io::Result<Vec<ClusterBatch>> {
    let contents = fs::read_to_string(plan_path)?;
    let batch_re =
        Regex::new(r"^#### Batch \d+: target `([^`]+)`").map_err(|err| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string())
        })?;
    let move_re = Regex::new(r"^- Move `([^`]+)` from `([^`]+)`").map_err(|err| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string())
    })?;
    let mut batches = Vec::new();
    let mut current: Option<ClusterBatch> = None;

    for line in contents.lines() {
        if let Some(captures) = batch_re.captures(line) {
            if let Some(batch) = current.take() {
                batches.push(batch);
            }
            current = Some(ClusterBatch {
                target: PathBuf::from(&captures[1]),
                moves: Vec::new(),
            });
            continue;
        }

        if let Some(captures) = move_re.captures(line) {
            if let Some(batch) = current.as_mut() {
                batch.moves.push(ClusterMove {
                    function: captures[1].to_string(),
                    from: PathBuf::from(&captures[2]),
                });
            }
        }
    }

    if let Some(batch) = current {
        batches.push(batch);
    }

    Ok(batches)
}

pub fn generate_phase2_cluster_slice(
    plan_path: &Path,
    batch_index: usize,
    root: &Path,
) -> std::io::Result<CorrectionIntelligenceReport> {
    let batches = parse_phase2_cluster_plan(plan_path)?;
    let batch = batches.get(batch_index.saturating_sub(1)).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Batch {} not found in {:?}", batch_index, plan_path),
        )
    })?;
    if batch.moves.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Batch {} in {:?} has no move entries",
                batch_index, plan_path
            ),
        ));
    }

    let mut plans = Vec::new();
    let mut policies = Vec::new();
    let mut criteria = Vec::new();
    let mut deltas = Vec::new();

    for entry in &batch.moves {
        let action = RefactorAction::MoveFunction {
            function: entry.function.clone(),
            from: entry.from.clone(),
            to: batch.target.clone(),
            required_layer: Some(batch.target.display().to_string()),
        };
        let predictions = vec![ViolationPrediction {
            violation_type: ViolationType::LayerViolation,
            affected_files: vec![entry.from.clone(), batch.target.clone()],
            severity: Severity::Medium,
            confidence: 0.6,
        }];
        let plan = generate_correction_plan(&action, &predictions);
        let policy = plan_verification_scope(&action, &plan);
        let rollback = build_rollback_criteria(&action, &plan);
        let delta = QualityDelta {
            action_id: plan.action_id.clone(),
            cohesion_delta: 0.0,
            violation_delta: 0,
            complexity_delta: 0.0,
            overall_score_delta: 0.0,
            acceptable: true,
            reason: "Not estimated for cluster slice".to_string(),
        };
        plans.push(plan);
        policies.push(policy);
        criteria.push(rollback);
        deltas.push(delta);
    }

    let summary = compute_summary(&plans, &deltas);

    Ok(CorrectionIntelligenceReport {
        version: "1.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        project_root: root.to_path_buf(),
        actions_analyzed: plans.len(),
        correction_plans: plans,
        verification_policies: policies,
        rollback_criteria: criteria,
        quality_deltas: deltas,
        summary,
    })
}

pub(crate) fn augment_path_coherence_strategies(
    plan: &mut CorrectionPlan,
    action: &RefactorAction,
    root: &Path,
) {
    let RefactorAction::RenameFile { from, to } = action else {
        return;
    };
    let Some(old_mod) = module_name_from_path(from) else {
        return;
    };
    let Some(new_mod) = module_name_from_path(to) else {
        return;
    };
    let old_file_name = from.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let new_file_name = to.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let replace_mod = old_mod != new_mod;

    let mod_re = if replace_mod {
        Regex::new(&format!(
            r"^\s*(pub\s+)?mod\s+{}\s*;",
            regex::escape(&old_mod)
        ))
        .ok()
    } else {
        None
    };
    let use_re = if replace_mod {
        Regex::new(&format!(
            r"^\s*use\s+.*\b{}\b",
            regex::escape(&old_mod)
        ))
        .ok()
    } else {
        None
    };
    let path_re = if !old_file_name.is_empty() && !new_file_name.is_empty() {
        Regex::new(r#"^\s*#\s*\[\s*path\s*=\s*"([^"]+)"\s*\]"#).ok()
    } else {
        None
    };

    let mut updates = Vec::new();
    let mut seen = HashSet::new();
    let rust_files = crate::cluster_010::gather_rust_files(root);

    for file in rust_files {
        let Ok(contents) = fs::read_to_string(&file) else {
            continue;
        };
        for line in contents.lines() {
            if let Some(re) = &mod_re {
                if re.is_match(line) {
                    let new_line = line.replace(&old_mod, &new_mod);
                    if new_line != line {
                        let key = (file.clone(), line.to_string(), new_line.clone());
                        if seen.insert(key.clone()) {
                            updates.push(key);
                        }
                    }
                    continue;
                }
            }
            if let Some(re) = &use_re {
                if re.is_match(line) {
                    let new_line = line.replace(&old_mod, &new_mod);
                    if new_line != line {
                        let key = (file.clone(), line.to_string(), new_line.clone());
                        if seen.insert(key.clone()) {
                            updates.push(key);
                        }
                    }
                    continue;
                }
            }
            if let Some(re) = &path_re {
                if re.is_match(line) && line.contains(old_file_name) {
                    let new_line = line.replace(old_file_name, new_file_name);
                    if new_line != line {
                        let key = (file.clone(), line.to_string(), new_line.clone());
                        if seen.insert(key.clone()) {
                            updates.push(key);
                        }
                    }
                }
            }
        }
    }

    updates.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for (file, old_ref, new_ref) in updates {
        plan.strategies.push(CorrectionStrategy::UpdateCaller {
            caller_file: file,
            old_ref,
            new_ref,
        });
    }
}

fn module_name_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    let name = if stem == "mod" {
        path.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())?
            .to_string()
    } else {
        stem.to_string()
    };
    Some(crate::cluster_010::normalize_module_name(&name))
}





pub(crate) fn compute_summary(plans: &[CorrectionPlan], deltas: &[QualityDelta]) -> CorrectionSummary {
    let mut trivial = 0;
    let mut moderate = 0;
    let mut complex = 0;
    let mut total_violations = 0;
    let mut total_confidence = 0.0;
    let mut total_time = 0;

    for plan in plans {
        match plan.tier {
            ErrorTier::Trivial => trivial += 1,
            ErrorTier::Moderate => moderate += 1,
            ErrorTier::Complex => complex += 1,
        }
        total_violations += plan.predicted_violations.len();
        total_confidence += plan.confidence;
        total_time += plan.estimated_fix_time_seconds;
    }

    let avg_conf = if plans.is_empty() {
        0.0
    } else {
        total_confidence / plans.len() as f64
    };

    let _ = deltas;

    CorrectionSummary {
        trivial_count: trivial,
        moderate_count: moderate,
        complex_count: complex,
        total_predicted_violations: total_violations,
        average_confidence: avg_conf,
        estimated_total_fix_time_seconds: total_time,
    }
}

pub(crate) fn fill_prediction_confidence(predictions: &mut [ViolationPrediction]) {
    for prediction in predictions {
        if prediction.confidence <= 0.0 {
            prediction.confidence = default_confidence(&prediction.violation_type);
        }
    }
}

fn default_confidence(violation_type: &crate::correction_plan_types::ViolationType) -> f64 {
    match violation_type {
        crate::correction_plan_types::ViolationType::UnresolvedImport => 0.95,
        crate::correction_plan_types::ViolationType::NameCollision => 1.0,
        crate::correction_plan_types::ViolationType::LayerViolation => 0.9,
        crate::correction_plan_types::ViolationType::VisibilityMismatch => 0.8,
        crate::correction_plan_types::ViolationType::BrokenReference => 0.85,
        crate::correction_plan_types::ViolationType::TypeMismatch => 0.6,
        crate::correction_plan_types::ViolationType::OwnershipIssue => 0.5,
    }
}

pub fn calculate_quality_delta(
    action: &RefactorAction,
    current: &Metrics,
    simulated: &Metrics,
) -> QualityDelta {
    let cohesion_delta = simulated.cohesion - current.cohesion;
    let violation_delta = simulated.violations as i32 - current.violations as i32;
    let complexity_delta = simulated.complexity - current.complexity;
    let overall = 0.5 * cohesion_delta - 0.3 * violation_delta as f64 - 0.2 * complexity_delta;
    let acceptable = overall > -0.05 && violation_delta <= 0;
    let reason = if acceptable {
        "Quality improved or maintained".to_string()
    } else if overall < -0.1 {
        "Quality degradation exceeds threshold".to_string()
    } else if violation_delta > 0 {
        format!("Introduced {} new violations", violation_delta)
    } else {
        "Quality barely acceptable".to_string()
    };
    QualityDelta {
        action_id: action.action_id(),
        cohesion_delta,
        violation_delta,
        complexity_delta,
        overall_score_delta: overall,
        acceptable,
        reason,
    }
}

pub(crate) fn action_function(action: &RefactorAction) -> Option<String> {
    match action {
        RefactorAction::MoveFunction { function, .. } => Some(function.clone()),
        _ => None,
    }
}

pub(crate) fn find_element_file(function: &str, elements: &[CodeElement]) -> Option<PathBuf> {
    elements
        .iter()
        .find(|el| el.name == function)
        .map(|el| PathBuf::from(&el.file_path))
}

pub(crate) fn symbol_exists(symbol: &str, elements: &[CodeElement]) -> bool {
    elements.iter().any(|el| el.name == symbol)
}

pub(crate) fn move_violates_invariant(
    _function: &str,
    _from: &PathBuf,
    _to: &PathBuf,
    _invariants: &InvariantAnalysisResult,
) -> bool {
    false
}

pub(crate) fn average_confidence(predictions: &[ViolationPrediction]) -> f64 {
    if predictions.is_empty() {
        return 1.0;
    }
    let total: f64 = predictions.iter().map(|p| p.confidence).sum();
    total / predictions.len() as f64
}

pub(crate) fn estimate_fix_time(count: usize) -> u32 {
    10 + (count as u32 * 5)
}

pub(crate) fn action_symbol(action: &RefactorAction) -> Option<String> {
    match action {
        RefactorAction::MoveFunction { function, .. } => Some(function.clone()),
        RefactorAction::RenameFunction { new_name, .. } => Some(new_name.clone()),
        RefactorAction::AdjustVisibility { symbol, .. } => Some(symbol.clone()),
        _ => None,
    }
}

pub(crate) fn action_module_path(action: &RefactorAction) -> String {
    match action {
        RefactorAction::MoveFunction { to, .. } => to.display().to_string(),
        RefactorAction::RenameFile { to, .. } => to.display().to_string(),
        RefactorAction::CreateFile { path } => path.display().to_string(),
        RefactorAction::AdjustVisibility { file, .. } => file.display().to_string(),
        _ => "crate".to_string(),
    }
}

pub(crate) fn action_refs(action: &RefactorAction) -> Option<(String, String)> {
    match action {
        RefactorAction::RenameFunction { old_name, new_name, .. } => {
            Some((old_name.clone(), new_name.clone()))
        }
        RefactorAction::RenameFile { from, to } => {
            Some((from.display().to_string(), to.display().to_string()))
        }
        _ => None,
    }
}

pub(crate) fn action_target_layer(action: &RefactorAction) -> Option<String> {
    match action {
        RefactorAction::MoveFunction { required_layer, .. } => required_layer.clone(),
        _ => None,
    }
}

pub(crate) fn action_visibility(
    action: &RefactorAction,
) -> Option<(
    String,
    std::path::PathBuf,
    crate::types::Visibility,
    crate::types::Visibility,
    String,
)> {
    match action {
        RefactorAction::AdjustVisibility {
            symbol,
            file,
            from,
            to,
            reason,
        } => Some((symbol.clone(), file.clone(), from.clone(), to.clone(), reason.clone())),
        _ => None,
    }
}

pub(crate) fn affected_files(action: &RefactorAction) -> Vec<std::path::PathBuf> {
    match action {
        RefactorAction::MoveFunction { from, to, .. } => vec![from.clone(), to.clone()],
        RefactorAction::RenameFunction { file, .. } => vec![file.clone()],
        RefactorAction::RenameFile { from, to } => vec![from.clone(), to.clone()],
        RefactorAction::CreateFile { path } => vec![path.clone()],
        RefactorAction::AdjustVisibility { file, .. } => vec![file.clone()],
    }
}

pub(crate) fn action_module(action: &RefactorAction) -> String {
    match action {
        RefactorAction::MoveFunction { to, .. } => to.display().to_string(),
        RefactorAction::RenameFunction { file, .. } => file.display().to_string(),
        RefactorAction::RenameFile { to, .. } => to.display().to_string(),
        RefactorAction::CreateFile { path } => path.display().to_string(),
        RefactorAction::AdjustVisibility { file, .. } => file.display().to_string(),
    }
}

pub(crate) fn estimate_verification_time(tier: &ErrorTier) -> u32 {
    match tier {
        ErrorTier::Trivial => 10,
        ErrorTier::Moderate => 60,
        ErrorTier::Complex => 180,
    }
}

pub(crate) fn extract_critical_tests(_action: &RefactorAction) -> Vec<String> {
    Vec::new()
}

pub(crate) fn find_callers(
    function: &str,
    call_graph: &HashMap<String, CallGraphNode>,
    elements: &[CodeElement],
) -> Vec<PathBuf> {
    let mut files = HashSet::new();
    if let Some(node) = call_graph.get(function) {
        for caller in &node.called_by {
            if let Some(file) = find_element_file(caller, elements) {
                files.insert(file);
            }
        }
    }
    files.into_iter().collect()
}

pub(crate) fn find_reference_files(
    function: &str,
    call_graph: &HashMap<String, CallGraphNode>,
    elements: &[CodeElement],
) -> Vec<PathBuf> {
    let mut files = HashSet::new();
    for (caller, node) in call_graph {
        if node.calls.iter().any(|c| c == function) {
            if let Some(file) = find_element_file(caller, elements) {
                files.insert(file);
            }
        }
    }
    files.into_iter().collect()
}

pub(crate) fn simulate_action(_action: &RefactorAction, state: &AnalysisState) -> AnalysisState {
    state.clone()
}

pub fn predict_violations(
    action: &RefactorAction,
    invariants: &InvariantAnalysisResult,
    call_graph: &HashMap<String, CallGraphNode>,
    elements: &[CodeElement],
) -> Vec<ViolationPrediction> {
    let mut predictions = Vec::new();
    match action {
        RefactorAction::MoveFunction { function, from, to, required_layer } => {
            let callers = find_callers(function, call_graph, elements);
            if !callers.is_empty() {
                predictions.push(ViolationPrediction {
                    violation_type: ViolationType::UnresolvedImport,
                    affected_files: callers,
                    severity: Severity::Critical,
                    confidence: 0.95,
                });
            }
            if let Some(layer) = required_layer {
                if !layer.is_empty() {
                    predictions.push(ViolationPrediction {
                        violation_type: ViolationType::LayerViolation,
                        affected_files: vec![to.clone()],
                        severity: Severity::High,
                        confidence: 1.0,
                    });
                }
            } else if move_violates_invariant(function, from, to, invariants) {
                predictions.push(ViolationPrediction {
                    violation_type: ViolationType::LayerViolation,
                    affected_files: vec![to.clone()],
                    severity: Severity::High,
                    confidence: 0.9,
                });
            }
        }
        RefactorAction::RenameFunction { old_name, new_name, file } => {
            if symbol_exists(new_name, elements) {
                predictions.push(ViolationPrediction {
                    violation_type: ViolationType::NameCollision,
                    affected_files: vec![file.clone()],
                    severity: Severity::Critical,
                    confidence: 1.0,
                });
            }
            let references = find_reference_files(old_name, call_graph, elements);
            if !references.is_empty() {
                predictions.push(ViolationPrediction {
                    violation_type: ViolationType::BrokenReference,
                    affected_files: references,
                    severity: Severity::Critical,
                    confidence: 0.85,
                });
            }
        }
        RefactorAction::RenameFile { from, to } => {
            predictions.push(ViolationPrediction {
                violation_type: ViolationType::BrokenReference,
                affected_files: vec![from.clone(), to.clone()],
                severity: Severity::High,
                confidence: 0.7,
            });
        }
        RefactorAction::CreateFile { path } => {
            predictions.push(ViolationPrediction {
                violation_type: ViolationType::UnresolvedImport,
                affected_files: vec![path.clone()],
                severity: Severity::Low,
                confidence: 0.5,
            });
        }
        RefactorAction::AdjustVisibility { file, .. } => {
            predictions.push(ViolationPrediction {
                violation_type: ViolationType::VisibilityMismatch,
                affected_files: vec![file.clone()],
                severity: Severity::Low,
                confidence: 0.8,
            });
        }
    }
    predictions
}

pub fn generate_correction_plan(
    action: &RefactorAction,
    predictions: &[ViolationPrediction],
) -> CorrectionPlan {
    let mut strategies = Vec::new();
    for prediction in predictions {
        match prediction.violation_type {
            ViolationType::UnresolvedImport => {
                if let Some(symbol) = action_symbol(action) {
                    strategies.push(CorrectionStrategy::AddImport {
                        module_path: action_module_path(action),
                        symbol,
                    });
                }
            }
            ViolationType::BrokenReference => {
                match action {
                    RefactorAction::RenameFile { .. } => {
                        if let Some((old_ref, new_ref)) = action_refs(action) {
                            strategies.push(CorrectionStrategy::UpdatePath {
                                old_path: old_ref,
                                new_path: new_ref,
                            });
                        }
                    }
                    _ => {
                        if let Some((old_ref, new_ref)) = action_refs(action) {
                            for file in &prediction.affected_files {
                                strategies.push(CorrectionStrategy::UpdateCaller {
                                    caller_file: file.clone(),
                                    old_ref: old_ref.clone(),
                                    new_ref: new_ref.clone(),
                                });
                            }
                        }
                    }
                }
            }
            ViolationType::NameCollision => {
                if let Some(symbol) = action_symbol(action) {
                    strategies.push(CorrectionStrategy::RenameWithSuffix {
                        original: symbol,
                        suffix: "_v2".to_string(),
                    });
                }
            }
            ViolationType::LayerViolation => {
                if let Some(layer) = action_target_layer(action) {
                    if let Some(function) = action_function(action) {
                        strategies.push(CorrectionStrategy::MoveToLayer {
                            function,
                            target_layer: layer,
                        });
                        if let Some(function) = action_function(action) {
                            if let Some(layer) = action_target_layer(action) {
                                strategies.push(CorrectionStrategy::EnsureImports {
                                    function,
                                    target_layer: layer,
                                });
                            }
                        }
                    }
                }
            }
            ViolationType::VisibilityMismatch => {
                if let Some((symbol, file, from, to, reason)) = action_visibility(action) {
                    if from == to || reason.starts_with("review:") {
                        let options = vec![
                            VisibilityPlanOption {
                                policy: "keep_public".to_string(),
                                target: crate::types::Visibility::Public,
                                requires_consent: false,
                                description: "Keep public (treat as external API).".to_string(),
                            },
                            VisibilityPlanOption {
                                policy: "downgrade_pub_crate".to_string(),
                                target: crate::types::Visibility::Crate,
                                requires_consent: true,
                                description:
                                    "Narrow to pub(crate) (internal API only).".to_string(),
                            },
                            VisibilityPlanOption {
                                policy: "downgrade_private".to_string(),
                                target: crate::types::Visibility::Private,
                                requires_consent: true,
                                description: "Narrow to private (file-local).".to_string(),
                            },
                        ];
                        strategies.push(CorrectionStrategy::VisibilityPlan {
                            symbol,
                            file,
                            current: from,
                            default_policy: "review_only".to_string(),
                            options,
                            notes: reason,
                        });
                    } else {
                        strategies.push(CorrectionStrategy::AdjustVisibility {
                            symbol,
                            file,
                            from,
                            to,
                            reason,
                        });
                    }
                }
            }
            ViolationType::TypeMismatch | ViolationType::OwnershipIssue => {
                strategies.push(CorrectionStrategy::ManualReview {
                    reason: format!("{:?} requires semantic analysis", prediction.violation_type),
                    context: format!("{:?}", action),
                });
            }
        }
    }

    let tier = predictions
        .iter()
        .map(classify_tier)
        .max()
        .unwrap_or(ErrorTier::Trivial);

    CorrectionPlan {
        action_id: action.action_id(),
        tier,
        predicted_violations: predictions.to_vec(),
        strategies,
        confidence: average_confidence(predictions),
        estimated_fix_time_seconds: estimate_fix_time(predictions.len()),
    }
}

pub fn plan_verification_scope(
    action: &RefactorAction,
    correction_plan: &CorrectionPlan,
) -> VerificationPolicy {
    let scope = match correction_plan.tier {
        ErrorTier::Trivial if correction_plan.predicted_violations.len() <= 3 => {
            VerificationScope::SyntaxOnly {
                files: affected_files(action),
            }
        }
        ErrorTier::Trivial | ErrorTier::Moderate => VerificationScope::ModuleLocal {
            module: action_module(action),
            transitive_depth: 2,
        },
        ErrorTier::Complex => VerificationScope::FullWorkspace,
    };

    let mut required_checks = vec![VerificationCheck::CargoCheck];
    if matches!(correction_plan.tier, ErrorTier::Moderate | ErrorTier::Complex) {
        required_checks.push(VerificationCheck::CargoTest { filter: None });
    }

    VerificationPolicy {
        action_id: correction_plan.action_id.clone(),
        scope,
        required_checks,
        incremental_eligible: matches!(correction_plan.tier, ErrorTier::Trivial),
        estimated_time_seconds: estimate_verification_time(&correction_plan.tier),
    }
}

pub fn build_rollback_criteria(
    action: &RefactorAction,
    correction_plan: &CorrectionPlan,
) -> RollbackCriteria {
    let mut mandatory = vec![RollbackCondition::BuildFailed];
    let mut suggested = vec![RollbackCondition::QualityDecreased { threshold: 0.05 }];

    match correction_plan.tier {
        ErrorTier::Complex => {
            mandatory.push(RollbackCondition::Tier3Error {
                error_type: ViolationType::TypeMismatch,
            });
            mandatory.push(RollbackCondition::ManualReviewRequired);
        }
        ErrorTier::Moderate => {
            suggested.push(RollbackCondition::TestsFailed {
                critical_tests: extract_critical_tests(action),
            });
        }
        ErrorTier::Trivial => {}
    }

    for prediction in &correction_plan.predicted_violations {
        if prediction.violation_type == ViolationType::LayerViolation {
            mandatory.push(RollbackCondition::InvariantViolated {
                invariant_ids: vec!["layer_ordering".to_string()],
            });
        }
    }

    RollbackCriteria {
        action_id: correction_plan.action_id.clone(),
        mandatory_rollback_if: mandatory,
        suggested_rollback_if: suggested,
    }
}

pub fn estimate_impact(action: &RefactorAction, current_state: &AnalysisState) -> QualityDelta {
    let simulated = simulate_action(action, current_state);
    calculate_quality_delta(action, &current_state.metrics, &simulated.metrics)
}

pub fn generate_intelligence_report(
    actions: &[RefactorAction],
    state: &IntelligenceState<'_>,
) -> CorrectionIntelligenceReport {
    let mut plans = Vec::new();
    let mut policies = Vec::new();
    let mut criteria = Vec::new();
    let mut deltas = Vec::new();

    for action in actions {
        let mut predictions =
            predict_violations(action, state.invariants, state.call_graph, state.elements);
        fill_prediction_confidence(&mut predictions);
        let mut plan = generate_correction_plan(action, &predictions);
        augment_path_coherence_strategies(&mut plan, action, &state.root);
        let policy = plan_verification_scope(action, &plan);
        let rollback = build_rollback_criteria(action, &plan);
        let delta = estimate_impact(action, &ImpactState {
            metrics: state.metrics.clone(),
        });

        plans.push(plan);
        policies.push(policy);
        criteria.push(rollback);
        deltas.push(delta);
    }

    let summary = compute_summary(&plans, &deltas);

    CorrectionIntelligenceReport {
        version: "1.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        project_root: state.root.clone(),
        actions_analyzed: actions.len(),
        correction_plans: plans,
        verification_policies: policies,
        rollback_criteria: criteria,
        quality_deltas: deltas,
        summary,
    }
}
