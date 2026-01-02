//! Refactoring constraints derived from invariants
//!
//! This module converts detected invariants into mechanical constraints that
//! enforce safe refactorings. Constraints are machine-readable rules that
//! determine what refactorings are allowed.

use crate::invariant_types::*;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A constraint that restricts refactoring operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefactorConstraint {
    /// Cannot move this item to a different file
    NoMove {
        target: String,
        reason: String,
        strength: InvariantStrength,
    },

    /// Cannot change the signature of this item
    PreserveSignature {
        target: String,
        signature: String,
        strength: InvariantStrength,
    },

    /// Cannot change the layer assignment
    FixedLayer {
        target: String,
        layer: usize,
        strength: InvariantStrength,
    },

    /// Must preserve ordering relative to other items
    PreserveOrdering {
        target: String,
        must_come_before: Vec<String>,
        strength: InvariantStrength,
    },

    /// Must preserve specific facts
    MustPreserve {
        target: String,
        facts: Vec<String>,
        strength: InvariantStrength,
    },

    /// Cannot delete this item (has critical dependencies)
    NoDelete {
        target: String,
        dependents: Vec<String>,
        strength: InvariantStrength,
    },

    /// Must maintain specific degree (in/out edges)
    PreserveDegree {
        target: String,
        in_degree: usize,
        out_degree: usize,
        strength: InvariantStrength,
    },
}

impl fmt::Display for RefactorConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefactorConstraint::NoMove { target, reason, strength } => {
                write!(f, "[{}] NoMove: {} (reason: {})", strength, target, reason)
            }
            RefactorConstraint::PreserveSignature { target, signature, strength } => {
                write!(f, "[{}] PreserveSignature: {} -> {}", strength, target, signature)
            }
            RefactorConstraint::FixedLayer { target, layer, strength } => {
                write!(f, "[{}] FixedLayer: {} @ layer {}", strength, target, layer)
            }
            RefactorConstraint::PreserveOrdering { target, must_come_before, strength } => {
                write!(f, "[{}] PreserveOrdering: {} before {:?}", strength, target, must_come_before)
            }
            RefactorConstraint::MustPreserve { target, facts, strength } => {
                write!(f, "[{}] MustPreserve: {} -> {} facts", strength, target, facts.len())
            }
            RefactorConstraint::NoDelete { target, dependents, strength } => {
                write!(f, "[{}] NoDelete: {} (dependents: {})", strength, target, dependents.len())
            }
            RefactorConstraint::PreserveDegree { target, in_degree, out_degree, strength } => {
                write!(f, "[{}] PreserveDegree: {} (in={}, out={})", strength, target, in_degree, out_degree)
            }
        }
    }
}

impl RefactorConstraint {
    /// Get the target of this constraint
    #[allow(dead_code)]
    pub fn target(&self) -> &str {
        match self {
            RefactorConstraint::NoMove { target, .. } => target,
            RefactorConstraint::PreserveSignature { target, .. } => target,
            RefactorConstraint::FixedLayer { target, .. } => target,
            RefactorConstraint::PreserveOrdering { target, .. } => target,
            RefactorConstraint::MustPreserve { target, .. } => target,
            RefactorConstraint::NoDelete { target, .. } => target,
            RefactorConstraint::PreserveDegree { target, .. } => target,
        }
    }

    /// Get the strength of the invariant backing this constraint
    #[allow(dead_code)]
    pub fn strength(&self) -> &InvariantStrength {
        match self {
            RefactorConstraint::NoMove { strength, .. } => strength,
            RefactorConstraint::PreserveSignature { strength, .. } => strength,
            RefactorConstraint::FixedLayer { strength, .. } => strength,
            RefactorConstraint::PreserveOrdering { strength, .. } => strength,
            RefactorConstraint::MustPreserve { strength, .. } => strength,
            RefactorConstraint::NoDelete { strength, .. } => strength,
            RefactorConstraint::PreserveDegree { strength, .. } => strength,
        }
    }

    /// Check if this constraint should block a refactoring
    #[allow(dead_code)]
    pub fn is_blocking(&self) -> bool {
        match self.strength() {
            InvariantStrength::Proven => true,
            InvariantStrength::Empirical { paths_checked } => *paths_checked >= 10,
            InvariantStrength::Heuristic => false,
        }
    }
}

/// Convert an invariant to a refactoring constraint
pub fn from_invariant(invariant: &Invariant) -> Option<RefactorConstraint> {
    match &invariant.kind {
        InvariantKind::Structural(s) => match s {
            StructuralInvariant::LayerFixed { layer } => Some(RefactorConstraint::FixedLayer {
                target: invariant.target.clone(),
                layer: *layer,
                strength: invariant.strength,
            }),
            StructuralInvariant::DegreeStable { in_degree, out_degree } => {
                Some(RefactorConstraint::PreserveDegree {
                    target: invariant.target.clone(),
                    in_degree: *in_degree,
                    out_degree: *out_degree,
                    strength: invariant.strength,
                })
            }
            StructuralInvariant::Leaf | StructuralInvariant::Root => {
                Some(RefactorConstraint::NoMove {
                    target: invariant.target.clone(),
                    reason: "graph topology fixed".to_string(),
                    strength: invariant.strength,
                })
            }
            StructuralInvariant::Bridge => Some(RefactorConstraint::NoDelete {
                target: invariant.target.clone(),
                dependents: vec!["graph connectivity".to_string()],
                strength: invariant.strength,
            }),
            StructuralInvariant::SccMembership { .. } => Some(RefactorConstraint::NoMove {
                target: invariant.target.clone(),
                reason: "SCC membership fixed".to_string(),
                strength: invariant.strength,
            }),
        },
        InvariantKind::Semantic(s) => match s {
            SemanticInvariant::TypeStable { signature } => {
                Some(RefactorConstraint::PreserveSignature {
                    target: invariant.target.clone(),
                    signature: signature.clone(),
                    strength: invariant.strength,
                })
            }
            SemanticInvariant::PureFunction | SemanticInvariant::Idempotent => {
                Some(RefactorConstraint::PreserveSignature {
                    target: invariant.target.clone(),
                    signature: "effects must remain pure".to_string(),
                    strength: invariant.strength,
                })
            }
            SemanticInvariant::EffectStable { .. } => Some(RefactorConstraint::PreserveSignature {
                target: invariant.target.clone(),
                signature: "effect signature fixed".to_string(),
                strength: invariant.strength,
            }),
        },
        InvariantKind::Delta(d) => match d {
            DeltaInvariant::Monotonic { .. } => Some(RefactorConstraint::PreserveOrdering {
                target: invariant.target.clone(),
                must_come_before: Vec::new(),
                strength: invariant.strength,
            }),
            _ => None, // Other delta invariants don't directly map to refactor constraints
        },
        InvariantKind::PathIntersection(p) => Some(RefactorConstraint::MustPreserve {
            target: invariant.target.clone(),
            facts: p.facts.iter().cloned().collect(),
            strength: invariant.strength,
        }),
    }
}

/// Check if a move operation is allowed by constraints
#[allow(dead_code)]
pub fn check_move_allowed(
    target: &str,
    current_file: &str,
    suggested_file: &str,
    constraints: &[RefactorConstraint],
) -> Result<(), String> {
    for constraint in constraints {
        if constraint.target() == target && constraint.is_blocking() {
            match constraint {
                RefactorConstraint::NoMove { reason, .. } => {
                    return Err(format!(
                        "Cannot move {} from {} to {}: {}",
                        target, current_file, suggested_file, reason
                    ));
                }
                RefactorConstraint::FixedLayer { layer, .. } => {
                    return Err(format!(
                        "Cannot move {} from {} to {}: layer {} is fixed",
                        target, current_file, suggested_file, layer
                    ));
                }
                _ => {} // Other constraints don't block moves
            }
        }
    }
    Ok(())
}

/// Generate all constraints from an invariant analysis result
pub fn generate_constraints(analysis: &InvariantAnalysisResult) -> Vec<RefactorConstraint> {
    analysis
        .invariants
        .iter()
        .filter_map(from_invariant)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_invariant_layer_fixed() {
        let inv = Invariant::new(
            "test_fn".to_string(),
            "test.rs".to_string(),
            InvariantKind::Structural(StructuralInvariant::LayerFixed { layer: 5 }),
            InvariantStrength::Proven,
            "Layer is fixed".to_string(),
        );

        let constraint = from_invariant(&inv).unwrap();
        match constraint {
            RefactorConstraint::FixedLayer { target, layer, .. } => {
                assert_eq!(target, "test_fn");
                assert_eq!(layer, 5);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_check_move_allowed_blocking() {
        let constraints = vec![RefactorConstraint::NoMove {
            target: "test_fn".to_string(),
            reason: "layer fixed".to_string(),
            strength: InvariantStrength::Proven,
        }];

        let result = check_move_allowed("test_fn", "old.rs", "new.rs", &constraints);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("layer fixed"));
    }

    #[test]
    fn test_check_move_allowed_non_blocking() {
        let constraints = vec![RefactorConstraint::NoMove {
            target: "test_fn".to_string(),
            reason: "heuristic".to_string(),
            strength: InvariantStrength::Heuristic,
        }];

        let result = check_move_allowed("test_fn", "old.rs", "new.rs", &constraints);
        assert!(result.is_ok()); // Heuristics don't block
    }

    #[test]
    fn test_constraint_is_blocking() {
        let proven = RefactorConstraint::NoMove {
            target: "fn1".to_string(),
            reason: "test".to_string(),
            strength: InvariantStrength::Proven,
        };
        assert!(proven.is_blocking());

        let heuristic = RefactorConstraint::NoMove {
            target: "fn2".to_string(),
            reason: "test".to_string(),
            strength: InvariantStrength::Heuristic,
        };
        assert!(!heuristic.is_blocking());
    }

    #[test]
    fn test_check_move_allowed() {
        let constraints = vec![RefactorConstraint::NoMove {
            target: "test_fn".to_string(),
            reason: "critical function".to_string(),
            strength: InvariantStrength::Proven,
        }];

        let result = check_move_allowed("test_fn", "a.rs", "b.rs", &constraints);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("critical function"));
    }
}
