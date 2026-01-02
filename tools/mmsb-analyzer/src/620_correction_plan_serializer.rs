#![allow(dead_code)]
/// Serialize correction plans to JSON values.
use std::path::Path;

use crate::correction_intelligence_report::CorrectionIntelligenceReport;
use crate::correction_plan_types::{CorrectionPlan, CorrectionStrategy};
use crate::quality_delta_types::RollbackCriteria;
use crate::verification_policy_emitter::emit_verification_policy;
use crate::verification_policy_types::{QualityThresholds, VerificationCheck, VerificationPolicy, VerificationScope};
use serde_json::{json, Value};

pub fn serialize_correction_plan(
    plan: &CorrectionPlan,
    verification: &VerificationPolicy,
    rollback: &RollbackCriteria,
) -> Value {
    json!({
        "action_id": plan.action_id,
        "tier": format!("{:?}", plan.tier),
        "confidence": plan.confidence,
        "estimated_fix_time_seconds": plan.estimated_fix_time_seconds,
        "predicted_violations": plan.predicted_violations.iter().map(|v| json!({
            "type": format!("{:?}", v.violation_type),
            "severity": format!("{:?}", v.severity),
            "affected_files": v.affected_files,
            "confidence": v.confidence,
        })).collect::<Vec<_>>(),
        "correction_strategies": plan.strategies.iter().map(serialize_strategy).collect::<Vec<_>>(),
        "verification_policy": {
            "scope": serialize_scope(&verification.scope),
            "required_checks": verification.required_checks.iter()
                .map(serialize_check)
                .collect::<Vec<_>>(),
            "incremental_eligible": verification.incremental_eligible,
            "estimated_time_seconds": verification.estimated_time_seconds,
        },
        "rollback_criteria": {
            "mandatory": rollback.mandatory_rollback_if.iter()
                .map(|c| format!("{:?}", c))
                .collect::<Vec<_>>(),
            "suggested": rollback.suggested_rollback_if.iter()
                .map(|c| format!("{:?}", c))
                .collect::<Vec<_>>(),
        }
    })
}

fn serialize_scope(scope: &VerificationScope) -> Value {
    match scope {
        VerificationScope::SyntaxOnly { files } => json!({
            "type": "SyntaxOnly",
            "files": files,
        }),
        VerificationScope::ModuleLocal { module, transitive_depth } => json!({
            "type": "ModuleLocal",
            "module": module,
            "transitive_depth": transitive_depth,
        }),
        VerificationScope::CallerChain { root_function } => json!({
            "type": "CallerChain",
            "root_function": root_function,
        }),
        VerificationScope::FullWorkspace => json!({
            "type": "FullWorkspace",
        }),
    }
}

fn serialize_check(check: &VerificationCheck) -> Value {
    match check {
        VerificationCheck::CargoCheck => json!({ "type": "CargoCheck" }),
        VerificationCheck::CargoTest { filter } => json!({
            "type": "CargoTest",
            "filter": filter,
        }),
        VerificationCheck::InvariantValidation { invariant_ids } => json!({
            "type": "InvariantValidation",
            "invariant_ids": invariant_ids,
        }),
        VerificationCheck::QualityMetrics { thresholds } => json!({
            "type": "QualityMetrics",
            "thresholds": serialize_thresholds(thresholds),
        }),
        VerificationCheck::ManualInspection { reason } => json!({
            "type": "ManualInspection",
            "reason": reason,
        }),
    }
}

fn serialize_thresholds(thresholds: &QualityThresholds) -> Value {
    json!({
        "min_cohesion_delta": thresholds.min_cohesion_delta,
        "max_violation_delta": thresholds.max_violation_delta,
        "max_complexity_delta": thresholds.max_complexity_delta,
    })
}

fn serialize_strategy(strategy: &CorrectionStrategy) -> Value {
    match strategy {
        CorrectionStrategy::AddImport { module_path, symbol } => json!({
            "type": "AddImport",
            "module_path": module_path,
            "symbol": symbol,
        }),
        CorrectionStrategy::UpdatePath { old_path, new_path } => json!({
            "type": "UpdatePath",
            "old_path": old_path,
            "new_path": new_path,
        }),
        CorrectionStrategy::ReExport { from_module, symbol } => json!({
            "type": "ReExport",
            "from_module": from_module,
            "symbol": symbol,
        }),
        CorrectionStrategy::RenameWithSuffix { original, suffix } => json!({
            "type": "RenameWithSuffix",
            "original": original,
            "suffix": suffix,
        }),
        CorrectionStrategy::MoveToLayer { function, target_layer } => json!({
            "type": "MoveToLayer",
            "function": function,
            "target_layer": target_layer,
        }),
        CorrectionStrategy::EnsureImports { function, target_layer } => json!({
            "type": "EnsureImports",
            "function": function,
            "target_layer": target_layer,
        }),
        CorrectionStrategy::UpdateCaller { caller_file, old_ref, new_ref } => json!({
            "type": "UpdateCaller",
            "caller_file": caller_file,
            "old_ref": old_ref,
            "new_ref": new_ref,
        }),
        CorrectionStrategy::AdjustVisibility {
            symbol,
            file,
            from,
            to,
            reason,
        } => json!({
            "type": "AdjustVisibility",
            "symbol": symbol,
            "file": file,
            "from": format!("{:?}", from),
            "to": format!("{:?}", to),
            "reason": reason,
        }),
        CorrectionStrategy::VisibilityPlan {
            symbol,
            file,
            current,
            default_policy,
            options,
            notes,
        } => json!({
            "type": "VisibilityPlan",
            "symbol": symbol,
            "file": file,
            "current": format!("{:?}", current),
            "default_policy": default_policy,
            "options": options.iter().map(|option| json!({
                "policy": option.policy,
                "target": format!("{:?}", option.target),
                "requires_consent": option.requires_consent,
                "description": option.description,
            })).collect::<Vec<_>>(),
            "notes": notes,
        }),
        CorrectionStrategy::ManualReview { reason, context } => json!({
            "type": "ManualReview",
            "reason": reason,
            "context": context,
        }),
    }
}

pub fn serialize_correction_plans(
    report: &CorrectionIntelligenceReport,
) -> serde_json::Value {
    let items = report
        .correction_plans
        .iter()
        .zip(report.verification_policies.iter())
        .zip(report.rollback_criteria.iter())
        .map(|((plan, policy), rollback)| serialize_correction_plan(plan, policy, rollback))
        .collect::<Vec<_>>();
    json!({
        "version": report.version,
        "timestamp": report.timestamp,
        "project_root": report.project_root,
        "actions_analyzed": report.actions_analyzed,
        "correction_plans": items,
        "quality_deltas": report.quality_deltas,
        "summary": report.summary,
    })
}

pub fn write_intelligence_outputs_at(
    report: &CorrectionIntelligenceReport,
    output_dir: &Path,
    correction_json: Option<&Path>,
    verification_policy_json: Option<&Path>,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let json_path = correction_json
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| output_dir.join("correction_intelligence.json"));
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let contract = serialize_correction_plans(report);
    std::fs::write(&json_path, serde_json::to_string_pretty(&contract)?)?;

    let policy_path = verification_policy_json
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| output_dir.join("verification_policy.json"));
    if let Some(parent) = policy_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    emit_verification_policy(&report.verification_policies, &policy_path)?;
    Ok(())
}

pub fn write_intelligence_outputs(
    report: &CorrectionIntelligenceReport,
    output_dir: &Path,
) -> std::io::Result<()> {
    write_intelligence_outputs_at(report, output_dir, None, None)
}
