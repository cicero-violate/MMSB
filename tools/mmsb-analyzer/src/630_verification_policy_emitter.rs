#![allow(dead_code)]
//! Emit verification policies to JSON.

use crate::verification_policy_types::{
    QualityThresholds, VerificationCheck, VerificationPolicy, VerificationScope,
};
use serde_json::json;
use std::path::Path;

pub fn emit_verification_policy(
    policies: &[VerificationPolicy],
    output_path: &Path,
) -> std::io::Result<()> {
    let policy_file = json!({
        "version": "1.0",
        "policies": policies.iter().map(|p| json!({
            "action_id": p.action_id,
            "scope": serialize_scope(&p.scope),
            "checks": p.required_checks.iter()
                .map(serialize_check)
                .collect::<Vec<_>>(),
            "incremental": p.incremental_eligible,
            "estimated_time_seconds": p.estimated_time_seconds,
        })).collect::<Vec<_>>()
    });
    std::fs::write(output_path, serde_json::to_string_pretty(&policy_file)?)?;
    Ok(())
}

fn serialize_scope(scope: &VerificationScope) -> serde_json::Value {
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

fn serialize_check(check: &VerificationCheck) -> serde_json::Value {
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

fn serialize_thresholds(thresholds: &QualityThresholds) -> serde_json::Value {
    json!({
        "min_cohesion_delta": thresholds.min_cohesion_delta,
        "max_violation_delta": thresholds.max_violation_delta,
        "max_complexity_delta": thresholds.max_complexity_delta,
    })
}


