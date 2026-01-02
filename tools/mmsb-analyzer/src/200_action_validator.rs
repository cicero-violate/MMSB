//! Action Validation
//!
//! Core validation logic for agent actions against refactoring constraints.
//! Matches proposed actions against mechanical constraints and generates violation reports.

use crate::refactor_constraints::RefactorConstraint;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Agent action proposal
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AgentAction {
    /// Move function to different file
    MoveFunction {
        name: String,
        from: PathBuf,
        to: PathBuf,
    },

    /// Rename function
    RenameFunction {
        old_name: String,
        new_name: String,
        file: PathBuf,
    },

    /// Change function signature
    ChangeSignature {
        name: String,
        old_sig: String,
        new_sig: String,
        file: PathBuf,
    },

    /// Delete function
    DeleteFunction { name: String, file: PathBuf },

    /// Create new function
    CreateFunction {
        name: String,
        file: PathBuf,
        signature: String,
    },
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationSeverity {
    Info,
    Warning,
    High,
    Critical,
}

/// Constraint violation with explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_id: usize,
    pub invariant_id: usize,
    pub reason: String,
    pub severity: ViolationSeverity,
    pub blocking: bool,
}

/// Extract layer number from file path (e.g., "src/040_test.rs" -> Some(40))
fn extract_layer(path: &PathBuf) -> Option<usize> {
    path.file_name()
        .and_then(|n| n.to_str())
        .and_then(|s| {
            let parts: Vec<&str> = s.split('_').collect();
            if !parts.is_empty() {
                parts[0].parse::<usize>().ok()
            } else {
                None
            }
        })
}

/// Validate action against all constraints
pub fn validate_action(
    action: &AgentAction,
    constraints: &[RefactorConstraint],
) -> Result<(), Vec<ConstraintViolation>> {
    let mut violations = Vec::new();

    for (idx, constraint) in constraints.iter().enumerate() {
        match (action, constraint) {
            // MoveFunction violations
            (
                AgentAction::MoveFunction { name, from, to },
                RefactorConstraint::NoMove {
                    target,
                    reason,
                    strength,
                },
            ) if target == name => {
                violations.push(ConstraintViolation {
                    constraint_id: idx,
                    invariant_id: 0, // No direct link to invariant
                    reason: format!("Cannot move {}: {} (strength: {:?})", name, reason, strength),
                    severity: ViolationSeverity::Critical,
                    blocking: true,
                });
            }

            (
                AgentAction::MoveFunction { name, from, to },
                RefactorConstraint::FixedLayer {
                    target,
                    layer,
                    strength,
                },
            ) if target == name => {
                let from_layer = extract_layer(from);
                let to_layer = extract_layer(to);

                if from_layer != to_layer {
                    violations.push(ConstraintViolation {
                        constraint_id: idx,
                        invariant_id: 0,
                        reason: format!(
                            "Cannot move {} across layers: layer {} fixed (strength: {:?})",
                            name, layer, strength
                        ),
                        severity: ViolationSeverity::Critical,
                        blocking: true,
                    });
                }
            }

            // ChangeSignature violations
            (
                AgentAction::ChangeSignature { name, old_sig, .. },
                RefactorConstraint::PreserveSignature {
                    target,
                    signature,
                    strength,
                },
            ) if target == name => {
                violations.push(ConstraintViolation {
                    constraint_id: idx,
                    invariant_id: 0,
                    reason: format!(
                        "Cannot change signature of {}: type-stable (strength: {:?})",
                        name, strength
                    ),
                    severity: ViolationSeverity::High,
                    blocking: true,
                });
            }

            // DeleteFunction violations
            (
                AgentAction::DeleteFunction { name, .. },
                RefactorConstraint::NoMove {
                    target,
                    reason,
                    strength,
                },
            ) if target == name && reason.contains("utility") => {
                // High in-degree functions shouldn't be deleted
                violations.push(ConstraintViolation {
                    constraint_id: idx,
                    invariant_id: 0,
                    reason: format!(
                        "Cannot delete {}: widely used utility function (strength: {:?})",
                        name, strength
                    ),
                    severity: ViolationSeverity::Critical,
                    blocking: true,
                });
            }

            _ => {}
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Check if a specific move is allowed
pub fn check_move_allowed(
    name: &str,
    from: &PathBuf,
    to: &PathBuf,
    constraints: &[RefactorConstraint],
) -> Result<(), String> {
    let action = AgentAction::MoveFunction {
        name: name.to_string(),
        from: from.clone(),
        to: to.clone(),
    };

    match validate_action(&action, constraints) {
        Ok(_) => Ok(()),
        Err(violations) => {
            let reasons: Vec<String> = violations.iter().map(|v| v.reason.clone()).collect();
            Err(reasons.join("; "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::invariant_types::InvariantStrength;

    #[test]
    fn test_extract_layer() {
        assert_eq!(extract_layer(&PathBuf::from("src/040_test.rs")), Some(40));
        assert_eq!(extract_layer(&PathBuf::from("src/000_utils.rs")), Some(0));
        assert_eq!(extract_layer(&PathBuf::from("src/test.rs")), None);
    }

    #[test]
    fn test_validate_no_move_constraint() {
        let constraints = vec![RefactorConstraint::NoMove {
            target: "test_fn".to_string(),
            reason: "layer 0 is fixed".to_string(),
            strength: InvariantStrength::Proven,
        }];

        let action = AgentAction::MoveFunction {
            name: "test_fn".to_string(),
            from: PathBuf::from("src/000_test.rs"),
            to: PathBuf::from("src/010_test.rs"),
        };

        let result = validate_action(&action, &constraints);
        assert!(result.is_err());

        let violations = result.unwrap_err();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].severity, ViolationSeverity::Critical);
        assert!(violations[0].blocking);
    }

    #[test]
    fn test_validate_layer_fixed_constraint() {
        let constraints = vec![RefactorConstraint::FixedLayer {
            target: "test_fn".to_string(),
            layer: 0,
            strength: InvariantStrength::Proven,
        }];

        let action = AgentAction::MoveFunction {
            name: "test_fn".to_string(),
            from: PathBuf::from("src/000_test.rs"),
            to: PathBuf::from("src/010_test.rs"),
        };

        let result = validate_action(&action, &constraints);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_preserve_signature() {
        let constraints = vec![RefactorConstraint::PreserveSignature {
            target: "test_fn".to_string(),
            signature: "fn test_fn() -> i32".to_string(),
            strength: InvariantStrength::Empirical { paths_checked: 5 },
        }];

        let action = AgentAction::ChangeSignature {
            name: "test_fn".to_string(),
            old_sig: "fn test_fn() -> i32".to_string(),
            new_sig: "fn test_fn() -> String".to_string(),
            file: PathBuf::from("src/test.rs"),
        };

        let result = validate_action(&action, &constraints);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_allowed_action() {
        let constraints = vec![RefactorConstraint::NoMove {
            target: "other_fn".to_string(),
            reason: "layer fixed".to_string(),
            strength: InvariantStrength::Proven,
        }];

        let action = AgentAction::MoveFunction {
            name: "test_fn".to_string(), // Different function
            from: PathBuf::from("src/000_test.rs"),
            to: PathBuf::from("src/010_test.rs"),
        };

        let result = validate_action(&action, &constraints);
        assert!(result.is_ok());
    }

}
