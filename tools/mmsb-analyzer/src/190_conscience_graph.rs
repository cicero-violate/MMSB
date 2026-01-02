//! Conscience Visualization
//!
//! Generates visual map of conscience: which functions are protected by invariants.

use crate::invariant_types::*;
use std::collections::HashMap;
use std::path::Path;

/// Generate conscience map showing protection levels per function
pub fn generate_conscience_map(
    invariants: &[Invariant],
    output_path: &Path,
) -> std::io::Result<()> {
    let mut content = String::new();

    content.push_str("# Conscience Map\n\n");
    content.push_str("## Overview\n\n");
    content.push_str("This map shows which functions are protected by mechanical constraints.\n");
    content.push_str("Functions with blocking invariants cannot be refactored without violating proven properties.\n\n");

    // Group invariants by function
    let mut by_function: HashMap<String, Vec<&Invariant>> = HashMap::new();

    for inv in invariants {
        by_function
            .entry(inv.target.clone())
            .or_default()
            .push(inv);
    }

    // Calculate statistics
    let total_functions = by_function.len();
    let protected_functions = by_function
        .values()
        .filter(|invs| invs.iter().any(|i| i.is_blocking()))
        .count();

    content.push_str(&format!(
        "**Total Functions**: {}\n\n",
        total_functions
    ));
    content.push_str(&format!(
        "**Protected Functions**: {} ({:.1}%)\n\n",
        protected_functions,
        (protected_functions as f64 / total_functions as f64) * 100.0
    ));

    // Sort by protection level (most protected first)
    let mut funcs: Vec<_> = by_function.into_iter().collect();
    funcs.sort_by_key(|(_, invs)| {
        -(invs.iter().filter(|i| i.is_blocking()).count() as i32)
    });

    content.push_str("---\n\n");
    content.push_str("## Functions by Protection Level\n\n");

    for (func, invs) in funcs {
        let blocking_count = invs.iter().filter(|i| i.is_blocking()).count();
        let total_count = invs.len();

        if blocking_count == 0 {
            continue; // Skip unprotected functions
        }

        let protection_percent = (blocking_count * 100) / total_count;

        content.push_str(&format!(
            "### `{}` ({}% protected)\n\n",
            func, protection_percent
        ));

        // Show file path if available
        if let Some(inv) = invs.first() {
            if !inv.file_path.is_empty() {
                content.push_str(&format!("**File**: `{}`\n\n", inv.file_path));
            }
        }

        // List blocking invariants
        for inv in invs.iter().filter(|i| i.is_blocking()) {
            content.push_str(&format!(
                "- 🔒 {} **{}**: {}\n",
                strength_emoji(inv),
                kind_name(inv),
                inv.description
            ));
        }

        content.push_str("\n");
    }

    // Legend
    content.push_str("---\n\n");
    content.push_str("## Legend\n\n");
    content.push_str("- ✓ **PROVEN**: Mathematical certainty from graph topology\n");
    content.push_str("- ◆ **EMPIRICAL**: Observed across multiple paths (high confidence)\n");
    content.push_str("- ? **HEURISTIC**: Name-based guess (LOW CONFIDENCE - verify manually)\n\n");
    content.push_str("- 🔒 **Blocking**: Constraint mechanically enforced\n\n");

    std::fs::write(output_path, content)?;
    Ok(())
}

/// Get emoji for invariant strength
fn strength_emoji(inv: &Invariant) -> &'static str {
    match inv.strength {
        InvariantStrength::Proven => "✓",
        InvariantStrength::Empirical { .. } => "◆",
        InvariantStrength::Heuristic => "?",
    }
}

/// Get short name for invariant kind
fn kind_name(inv: &Invariant) -> String {
    match &inv.kind {
        InvariantKind::Structural(s) => match s {
            StructuralInvariant::LayerFixed { layer } => format!("LayerFixed({})", layer),
            StructuralInvariant::DegreeStable { in_degree, out_degree } => {
                format!("DegreeStable(in={}, out={})", in_degree, out_degree)
            }
            StructuralInvariant::Leaf => "Leaf".to_string(),
            StructuralInvariant::Root => "Root".to_string(),
            StructuralInvariant::Bridge => "Bridge".to_string(),
            StructuralInvariant::SccMembership { scc_id, scc_size } => {
                format!("SCC({}, size={})", scc_id, scc_size)
            }
        },
        InvariantKind::Semantic(s) => match s {
            SemanticInvariant::TypeStable { .. } => "TypeStable".to_string(),
            SemanticInvariant::PureFunction => "PureFunction".to_string(),
            SemanticInvariant::Idempotent => "Idempotent".to_string(),
            SemanticInvariant::EffectStable { .. } => "EffectStable".to_string(),
        },
        InvariantKind::Delta(_) => "Delta".to_string(),
        InvariantKind::PathIntersection { .. } => "PathIntersection".to_string(),
    }
}

/// Generate summary statistics
#[allow(dead_code)]
pub fn generate_conscience_stats(invariants: &[Invariant]) -> ConscienceStats {
    let mut by_function: HashMap<String, Vec<&Invariant>> = HashMap::new();

    for inv in invariants {
        by_function
            .entry(inv.target.clone())
            .or_default()
            .push(inv);
    }

    let total_functions = by_function.len();
    let protected_functions = by_function
        .values()
        .filter(|invs| invs.iter().any(|i| i.is_blocking()))
        .count();

    let proven_count = invariants
        .iter()
        .filter(|i| matches!(i.strength, InvariantStrength::Proven))
        .count();

    let empirical_count = invariants
        .iter()
        .filter(|i| matches!(i.strength, InvariantStrength::Empirical { .. }))
        .count();

    let heuristic_count = invariants
        .iter()
        .filter(|i| matches!(i.strength, InvariantStrength::Heuristic))
        .count();

    ConscienceStats {
        total_functions,
        protected_functions,
        total_invariants: invariants.len(),
        blocking_invariants: invariants.iter().filter(|i| i.is_blocking()).count(),
        proven_count,
        empirical_count,
        heuristic_count,
    }
}

#[allow(dead_code)]
pub struct ConscienceStats {
    pub total_functions: usize,
    pub protected_functions: usize,
    pub total_invariants: usize,
    pub blocking_invariants: usize,
    pub proven_count: usize,
    pub empirical_count: usize,
    pub heuristic_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_invariant(
        name: &str,
        kind: InvariantKind,
        strength: InvariantStrength,
    ) -> Invariant {
        Invariant::new(
            name.to_string(),
            "test.rs".to_string(),
            kind,
            strength,
            "test".to_string(),
        )
    }

    #[test]
    fn test_generate_stats() {
        let invariants = vec![
            make_test_invariant(
                "fn1",
                InvariantKind::Structural(StructuralInvariant::LayerFixed { layer: 0 }),
                InvariantStrength::Proven,
            ),
            make_test_invariant(
                "fn1",
                InvariantKind::Semantic(SemanticInvariant::PureFunction),
                InvariantStrength::Heuristic,
            ),
            make_test_invariant(
                "fn2",
                InvariantKind::Structural(StructuralInvariant::Leaf),
                InvariantStrength::Proven,
            ),
        ];

        let stats = generate_conscience_stats(&invariants);

        assert_eq!(stats.total_functions, 2); // fn1 and fn2
        assert_eq!(stats.total_invariants, 3);
        assert_eq!(stats.proven_count, 2);
        assert_eq!(stats.heuristic_count, 1);
    }

    #[test]
    fn test_strength_emoji() {
        let proven = make_test_invariant(
            "test",
            InvariantKind::Structural(StructuralInvariant::Leaf),
            InvariantStrength::Proven,
        );
        let empirical = make_test_invariant(
            "test",
            InvariantKind::Semantic(SemanticInvariant::TypeStable {
                signature: "sig".to_string(),
            }),
            InvariantStrength::Empirical { paths_checked: 5 },
        );
        let heuristic = make_test_invariant(
            "test",
            InvariantKind::Semantic(SemanticInvariant::PureFunction),
            InvariantStrength::Heuristic,
        );

        assert_eq!(strength_emoji(&proven), "✓");
        assert_eq!(strength_emoji(&empirical), "◆");
        assert_eq!(strength_emoji(&heuristic), "?");
    }
}
