//! Invariant Reporting
//!
//! This module generates reports for detected invariants, exports them to JSON,
//! and integrates with the existing reporting system.

use crate::invariant_types::*;
use crate::refactor_constraints::RefactorConstraint;
use serde_json;
use std::fs;
use std::path::Path;

/// Generate invariant report in markdown format
pub fn generate_invariant_report(
    result: &InvariantAnalysisResult,
    output_dir: &Path,
) -> Result<(), std::io::Error> {
    let report_dir = output_dir.join("95_invariants");
    fs::create_dir_all(&report_dir)?;

    let report_path = report_dir.join("index.md");
    let mut report = String::new();

    // Header
    report.push_str("# Invariant Analysis Report\n\n");
    report.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    // Summary
    report.push_str("## Summary\n\n");
    report.push_str(&format!("- **Total Invariants**: {}\n", result.stats.total_count));
    report.push_str(&format!("- **Proven**: {} ({:.1}%)\n",
        result.stats.proven_count,
        result.stats.proven_percentage()
    ));
    report.push_str(&format!("- **Empirical**: {}\n", result.stats.empirical_count));
    report.push_str(&format!("- **Heuristic**: {} ({:.1}%) ⚠️ LOW CONFIDENCE\n",
        result.stats.heuristic_count,
        result.stats.heuristic_percentage()
    ));
    report.push_str(&format!("- **Violations**: {}\n\n", result.stats.violation_count));

    report.push_str("### By Kind\n\n");
    report.push_str(&format!("- **Structural**: {}\n", result.stats.structural_count));
    report.push_str(&format!("- **Semantic**: {}\n", result.stats.semantic_count));
    report.push_str(&format!("- **Delta**: {}\n", result.stats.delta_count));
    report.push_str(&format!("- **Path-Intersection**: {}\n\n", result.stats.path_intersection_count));

    // Proven Invariants
    report.push_str("## Proven Invariants (Mechanical Truth)\n\n");
    report.push_str("These invariants are mathematically proven from graph structure and should **always block refactorings**.\n\n");

    let proven: Vec<_> = result
        .invariants
        .iter()
        .filter(|inv| matches!(inv.strength, InvariantStrength::Proven))
        .collect();

    if proven.is_empty() {
        report.push_str("*None detected*\n\n");
    } else {
        for inv in &proven {
            report.push_str(&format!("### {}\n\n", inv.target));
            report.push_str(&format!("- **Type**: {}\n", inv.kind));
            report.push_str(&format!("- **File**: {}\n", inv.file_path));
            report.push_str(&format!("- **Description**: {}\n", inv.description));
            if !inv.evidence.is_empty() {
                report.push_str("- **Evidence**:\n");
                for e in &inv.evidence {
                    report.push_str(&format!("  - {}\n", e));
                }
            }
            report.push_str("\n");
        }
    }

    // Empirical Invariants
    report.push_str("## Empirical Invariants (High Confidence)\n\n");
    report.push_str("These invariants were observed across multiple paths/samples and have high confidence.\n\n");

    let empirical: Vec<_> = result
        .invariants
        .iter()
        .filter(|inv| matches!(inv.strength, InvariantStrength::Empirical { .. }))
        .collect();

    if empirical.is_empty() {
        report.push_str("*None detected*\n\n");
    } else {
        for inv in empirical.iter().take(20) {
            // Limit to first 20 for readability
            report.push_str(&format!("### {}\n\n", inv.target));
            report.push_str(&format!("- **Type**: {}\n", inv.kind));
            report.push_str(&format!("- **Strength**: {}\n", inv.strength));
            report.push_str(&format!("- **Confidence**: {:.2}\n", inv.confidence.value()));
            report.push_str(&format!("- **Description**: {}\n\n", inv.description));
        }
        if empirical.len() > 20 {
            report.push_str(&format!("*... and {} more*\n\n", empirical.len() - 20));
        }
    }

    // Heuristic Signals
    report.push_str("## Heuristic Signals (Low Confidence - Review Required)\n\n");
    report.push_str("⚠️ **WARNING**: These are based on naming patterns and heuristics. They require manual verification and should **NOT block refactorings**.\n\n");

    let heuristic: Vec<_> = result
        .invariants
        .iter()
        .filter(|inv| matches!(inv.strength, InvariantStrength::Heuristic))
        .collect();

    if heuristic.is_empty() {
        report.push_str("*None detected*\n\n");
    } else {
        for inv in heuristic.iter().take(10) {
            // Limit to first 10
            report.push_str(&format!("- **{}**: {} ({})\n", inv.target, inv.description, inv.file_path));
        }
        if heuristic.len() > 10 {
            report.push_str(&format!("\n*... and {} more (see JSON export)*\n\n", heuristic.len() - 10));
        } else {
            report.push_str("\n");
        }
    }

    // Violations
    if !result.violations.is_empty() {
        report.push_str("## Violations\n\n");
        report.push_str("Detected violations of invariants, grouped by severity.\n\n");

        let mut critical: Vec<_> = result.violations.iter()
            .filter(|v| matches!(v.severity, ViolationSeverity::Critical))
            .collect();
        critical.sort_by_key(|v| &v.invariant.target);

        if !critical.is_empty() {
            report.push_str("### Critical\n\n");
            for violation in critical {
                report.push_str(&format!("- **{}**: {}\n",
                    violation.invariant.target,
                    violation.violation_description
                ));
                if let Some(ref fix) = violation.suggested_fix {
                    report.push_str(&format!("  - *Suggested fix*: {}\n", fix));
                }
            }
            report.push_str("\n");
        }
    }

    // Layer Assignments
    if !result.layer_assignments.is_empty() {
        report.push_str("## Layer Assignments (Inferred from Call Graph)\n\n");
        report.push_str("Layers are **NOT** based on filename prefixes. They are computed from the call graph structure.\n\n");

        let mut layers: Vec<_> = result.layer_assignments.iter().collect();
        layers.sort_by_key(|(_, info)| info.layer);

        for (name, info) in layers.iter().take(20) {
            report.push_str(&format!("- **{}**: Layer {} (dependencies: {})\n",
                name,
                info.layer,
                info.dependencies.len()
            ));
        }
        if layers.len() > 20 {
            report.push_str(&format!("\n*... and {} more (see JSON export)*\n\n", layers.len() - 20));
        } else {
            report.push_str("\n");
        }
    }

    fs::write(&report_path, report)?;
    println!("📄 Invariant report written to: {}", report_path.display());

    // Export JSON
    export_json(result, &report_dir)?;

    // Generate conscience map
    let conscience_map_path = report_dir.join("conscience_map.md");
    crate::conscience_graph::generate_conscience_map(&result.invariants, &conscience_map_path)?;
    println!("📄 Conscience map written to: {}", conscience_map_path.display());

    Ok(())
}

/// Export invariants to JSON for agent consumption
pub fn export_json(
    result: &InvariantAnalysisResult,
    output_dir: &Path,
) -> Result<(), std::io::Error> {
    let json_path = output_dir.join("invariants.json");
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(&json_path, json)?;
    println!("📄 JSON export written to: {}", json_path.display());
    Ok(())
}

/// Export refactoring constraints to JSON
pub fn export_constraints_json(
    constraints: &[RefactorConstraint],
    output_dir: &Path,
) -> Result<(), std::io::Error> {
    let constraints_dir = output_dir.join("96_constraints");
    fs::create_dir_all(&constraints_dir)?;

    let json_path = constraints_dir.join("refactor_constraints.json");
    let json = serde_json::to_string_pretty(constraints)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(&json_path, json)?;
    println!("📄 Constraints written to: {}", json_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_generate_report() {
        let result = InvariantAnalysisResult {
            invariants: vec![
                Invariant::new(
                    "test_fn".to_string(),
                    "test.rs".to_string(),
                    InvariantKind::Structural(StructuralInvariant::Leaf),
                    InvariantStrength::Proven,
                    "Test invariant".to_string(),
                ),
            ],
            violations: Vec::new(),
            layer_assignments: HashMap::new(),
            stats: InvariantStats {
                total_count: 1,
                proven_count: 1,
                empirical_count: 0,
                heuristic_count: 0,
                structural_count: 1,
                semantic_count: 0,
                delta_count: 0,
                path_intersection_count: 0,
                violation_count: 0,
            },
        };

        let temp_dir = std::env::temp_dir().join("mmsb_test_invariants");
        fs::create_dir_all(&temp_dir).unwrap();

        let res = generate_invariant_report(&result, &temp_dir);
        assert!(res.is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
