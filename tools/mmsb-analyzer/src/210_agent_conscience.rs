//! Agent Conscience
//!
//! Conscience-driven agent architecture: Validates actions against invariants.
//! Philosophy: An agent with "conscience" mechanically enforces invariants.
//! No interpretation, no values - just constraint checking.

use crate::action_validator::{validate_action, AgentAction, ConstraintViolation};
use crate::invariant_types::{Invariant, InvariantStrength};
use crate::refactor_constraints::{from_invariant, RefactorConstraint};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Action permission result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPermission {
    pub allowed: bool,
    pub action: AgentAction,
    pub violations: Vec<ConstraintViolation>,
    pub warnings: Vec<String>,
}

/// Agent conscience: Validates actions against invariants
pub struct AgentConscience {
    invariants: Vec<Invariant>,
    constraints: Vec<RefactorConstraint>,
}

impl AgentConscience {
    /// Create conscience from detected invariants
    pub fn new(invariants: Vec<Invariant>) -> Self {
        let constraints = invariants
            .iter()
            .filter_map(from_invariant)
            .collect();

        Self {
            invariants,
            constraints,
        }
    }

    /// Load conscience from JSON file
    #[allow(dead_code)]
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;

        #[derive(Deserialize)]
        struct ConscienceExport {
            invariants: Vec<Invariant>,
        }

        let export: ConscienceExport = serde_json::from_str(&json)?;
        Ok(Self::new(export.invariants))
    }

    /// Check if action is morally permissible (preserves invariants)
    ///
    /// This is the humility gate: H(A, I) = ⊤ iff ∀I_k ∈ I : ¬violates(A, I_k)
    pub fn check_action(&self, action: &AgentAction) -> ActionPermission {
        match validate_action(action, &self.constraints) {
            Ok(_) => ActionPermission {
                allowed: true,
                action: action.clone(),
                violations: Vec::new(),
                warnings: self.generate_warnings(action),
            },
            Err(violations) => ActionPermission {
                allowed: false,
                action: action.clone(),
                violations,
                warnings: Vec::new(),
            },
        }
    }

    /// Generate warnings for non-blocking concerns
    fn generate_warnings(&self, action: &AgentAction) -> Vec<String> {
        let mut warnings = Vec::new();

        // Warn about heuristic invariants
        for inv in &self.invariants {
            if matches!(inv.strength, InvariantStrength::Heuristic) {
                if let Some(warning) = self.check_heuristic_warning(action, inv) {
                    warnings.push(warning);
                }
            }
        }

        warnings
    }

    /// Check if heuristic invariant suggests caution
    fn check_heuristic_warning(&self, action: &AgentAction, inv: &Invariant) -> Option<String> {
        use crate::invariant_types::{InvariantKind, SemanticInvariant};

        match (&action, &inv.kind) {
            (
                AgentAction::MoveFunction { name, .. },
                InvariantKind::Semantic(SemanticInvariant::PureFunction),
            ) if inv.target == *name => {
                Some(format!(
                    "Warning: {} may be pure function (HEURISTIC - verify manually)",
                    name
                ))
            }
            (
                AgentAction::ChangeSignature { name, .. },
                InvariantKind::Semantic(SemanticInvariant::Idempotent),
            ) if inv.target == *name => {
                Some(format!(
                    "Warning: {} may be idempotent (HEURISTIC - verify manually)",
                    name
                ))
            }
            _ => None,
        }
    }

    /// Query: "What can I safely do to this function?"
    pub fn query_allowed_actions(&self, function: &str) -> Vec<AgentAction> {
        let mut allowed = Vec::new();

        // Test rename (usually safe if no signature change)
        let rename_action = AgentAction::RenameFunction {
            old_name: function.to_string(),
            new_name: format!("{}_renamed", function),
            file: PathBuf::from("test.rs"),
        };

        if self.check_action(&rename_action).allowed {
            allowed.push(rename_action);
        }

        // Test creation (always safe)
        allowed.push(AgentAction::CreateFunction {
            name: format!("new_{}", function),
            file: PathBuf::from("test.rs"),
            signature: "fn new() -> ()".to_string(),
        });

        allowed
    }

    /// Export conscience as JSON (for agents to load)
    #[allow(dead_code)]
    pub fn export_json(&self, path: &Path) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct ConscienceExport {
            invariants: Vec<Invariant>,
            constraints: Vec<RefactorConstraint>,
            meta: ConscMeta,
        }

        #[derive(Serialize)]
        struct ConscMeta {
            total_invariants: usize,
            blocking_invariants: usize,
            total_constraints: usize,
        }

        let export = ConscienceExport {
            invariants: self.invariants.clone(),
            constraints: self.constraints.clone(),
            meta: ConscMeta {
                total_invariants: self.invariants.len(),
                blocking_invariants: self.invariants.iter().filter(|i| i.is_blocking()).count(),
                total_constraints: self.constraints.len(),
            },
        };

        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Get statistics about conscience state
    #[allow(dead_code)]
    pub fn stats(&self) -> ConscienceStats {
        let blocking = self.invariants.iter().filter(|i| i.is_blocking()).count();
        let proven = self
            .invariants
            .iter()
            .filter(|i| matches!(i.strength, InvariantStrength::Proven))
            .count();
        let empirical = self
            .invariants
            .iter()
            .filter(|i| matches!(i.strength, InvariantStrength::Empirical { .. }))
            .count();
        let heuristic = self
            .invariants
            .iter()
            .filter(|i| matches!(i.strength, InvariantStrength::Heuristic))
            .count();

        ConscienceStats {
            total_invariants: self.invariants.len(),
            blocking_invariants: blocking,
            total_constraints: self.constraints.len(),
            proven_count: proven,
            empirical_count: empirical,
            heuristic_count: heuristic,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ConscienceStats {
    pub total_invariants: usize,
    pub blocking_invariants: usize,
    pub total_constraints: usize,
    pub proven_count: usize,
    pub empirical_count: usize,
    pub heuristic_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariant_types::{InvariantKind, StructuralInvariant};

    fn make_test_invariant(name: &str, layer: usize, strength: InvariantStrength) -> Invariant {
        Invariant::new(
            name.to_string(),
            format!("src/{:03}_test.rs", layer * 10),
            InvariantKind::Structural(StructuralInvariant::LayerFixed { layer }),
            strength,
            format!("Layer {} fixed", layer),
        )
    }

    #[test]
    fn test_conscience_blocks_invalid_move() {
        let inv = make_test_invariant("test_fn", 0, InvariantStrength::Proven);
        let conscience = AgentConscience::new(vec![inv]);

        let action = AgentAction::MoveFunction {
            name: "test_fn".to_string(),
            from: PathBuf::from("src/000_test.rs"),
            to: PathBuf::from("src/010_test.rs"),
        };

        let result = conscience.check_action(&action);
        assert!(!result.allowed);
        assert!(!result.violations.is_empty());
    }

    #[test]
    fn test_conscience_allows_valid_action() {
        let inv = make_test_invariant("other_fn", 0, InvariantStrength::Proven);
        let conscience = AgentConscience::new(vec![inv]);

        let action = AgentAction::MoveFunction {
            name: "test_fn".to_string(), // Different function
            from: PathBuf::from("src/000_test.rs"),
            to: PathBuf::from("src/010_test.rs"),
        };

        let result = conscience.check_action(&action);
        assert!(result.allowed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_query_allowed_actions() {
        let inv = make_test_invariant("test_fn", 0, InvariantStrength::Proven);
        let conscience = AgentConscience::new(vec![inv]);

        let allowed = conscience.query_allowed_actions("test_fn");
        assert!(!allowed.is_empty()); // Should return at least some safe actions
    }

    #[test]
    fn test_conscience_stats() {
        let invariants = vec![
            make_test_invariant("fn1", 0, InvariantStrength::Proven),
            make_test_invariant("fn2", 1, InvariantStrength::Empirical { paths_checked: 3 }),
            make_test_invariant("fn3", 2, InvariantStrength::Heuristic),
        ];

        let conscience = AgentConscience::new(invariants);
        let stats = conscience.stats();

        assert_eq!(stats.total_invariants, 3);
        assert_eq!(stats.proven_count, 1);
        assert_eq!(stats.empirical_count, 1);
        assert_eq!(stats.heuristic_count, 1);
    }

}
