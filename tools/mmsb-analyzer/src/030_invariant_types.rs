//! Core types for invariant detection and analysis
//!
//! This module defines the fundamental data structures for detecting and representing
//! invariants in code. Invariants are properties that remain stable across refactorings
//! and serve as constraints for automated code transformations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Strength classification for invariants
///
/// CRITICAL: Heuristics are NOT invariants - they are signals that need verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvariantStrength {
    /// Mathematically proven from graph structure (e.g., SCC membership, degree)
    Proven,

    /// Observed empirically across many paths/samples with high confidence
    Empirical { paths_checked: usize },

    /// Name-based or keyword detection - LOW confidence, needs review
    Heuristic,
}

impl fmt::Display for InvariantStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvariantStrength::Proven => write!(f, "PROVEN"),
            InvariantStrength::Empirical { paths_checked } => {
                write!(f, "EMPIRICAL (paths: {})", paths_checked)
            }
            InvariantStrength::Heuristic => write!(f, "HEURISTIC (low confidence)"),
        }
    }
}

/// Confidence score for an invariant [0.0, 1.0]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(pub f64);

impl Confidence {
    pub fn from_strength(strength: &InvariantStrength) -> Self {
        match strength {
            InvariantStrength::Proven => Confidence(1.0),
            InvariantStrength::Empirical { paths_checked } => {
                if *paths_checked > 100 {
                    Confidence(0.9)
                } else if *paths_checked > 10 {
                    Confidence(0.7)
                } else {
                    Confidence(0.5)
                }
            }
            InvariantStrength::Heuristic => Confidence(0.3), // Always low!
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Structural invariants - properties derived from graph topology
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuralInvariant {
    /// Layer assignment is fixed and cannot change
    LayerFixed { layer: usize },

    /// In-degree and out-degree are stable
    DegreeStable { in_degree: usize, out_degree: usize },

    /// Node is a leaf (out-degree = 0)
    Leaf,

    /// Node is a root (in-degree = 0)
    Root,

    /// Node is a bridge (removal disconnects graph)
    Bridge,

    /// Node belongs to a specific SCC
    SccMembership { scc_id: usize, scc_size: usize },
}

/// Semantic invariants - properties of function behavior
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticInvariant {
    /// Function signature is stable (type-based)
    TypeStable { signature: String },

    /// Function is pure (no side effects, deterministic)
    PureFunction,

    /// Function is idempotent: f(f(x)) = f(x)
    Idempotent,

    /// Function has specific effect signature
    EffectStable { effects: Vec<String> },
}

/// Delta invariants - properties about rate of change
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeltaInvariant {
    /// Value is monotonically increasing/decreasing
    Monotonic { direction: MonotonicDirection },

    /// Change rate is bounded
    Stable { epsilon: f64 },

    /// Only appends, never deletes
    AppendOnly,

    /// Sum of inputs equals sum of outputs
    Conservative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonotonicDirection {
    Increasing,
    Decreasing,
}

/// Path-intersection invariant - facts that hold on ALL execution paths
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PathIntersectionInvariant {
    /// Symbolic facts that hold on all paths
    pub facts: HashSet<String>,

    /// Number of paths analyzed
    pub paths_analyzed: usize,

    /// Maximum path depth explored
    pub max_depth: usize,
}

/// Complete invariant kind taxonomy
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InvariantKind {
    Structural(StructuralInvariant),
    Semantic(SemanticInvariant),
    Delta(DeltaInvariant),
    PathIntersection(PathIntersectionInvariant),
}

impl fmt::Display for InvariantKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvariantKind::Structural(s) => write!(f, "Structural: {:?}", s),
            InvariantKind::Semantic(s) => write!(f, "Semantic: {:?}", s),
            InvariantKind::Delta(d) => write!(f, "Delta: {:?}", d),
            InvariantKind::PathIntersection(p) => {
                write!(f, "PathIntersection: {} facts across {} paths",
                       p.facts.len(), p.paths_analyzed)
            }
        }
    }
}

/// A detected invariant with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Invariant {
    /// Target of the invariant (function name, module name, etc.)
    pub target: String,

    /// File where the target is located
    pub file_path: String,

    /// Kind of invariant
    pub kind: InvariantKind,

    /// Strength classification
    pub strength: InvariantStrength,

    /// Confidence score
    pub confidence: Confidence,

    /// Human-readable description
    pub description: String,

    /// Evidence supporting this invariant
    pub evidence: Vec<String>,
}

impl Invariant {
    /// Create a new invariant
    pub fn new(
        target: String,
        file_path: String,
        kind: InvariantKind,
        strength: InvariantStrength,
        description: String,
    ) -> Self {
        let confidence = Confidence::from_strength(&strength);
        Self {
            target,
            file_path,
            kind,
            strength,
            confidence,
            description,
            evidence: Vec::new(),
        }
    }

    /// Add evidence supporting this invariant
    pub fn add_evidence(&mut self, evidence: String) {
        self.evidence.push(evidence);
    }

    /// Check if this invariant should block refactorings
    pub fn is_blocking(&self) -> bool {
        match self.strength {
            InvariantStrength::Proven => true,
            InvariantStrength::Empirical { paths_checked } => paths_checked >= 10,
            InvariantStrength::Heuristic => false, // Heuristics never block
        }
    }
}

/// A violation of a detected invariant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantViolation {
    /// The invariant that was violated
    pub invariant: Invariant,

    /// Description of the violation
    pub violation_description: String,

    /// Severity level
    pub severity: ViolationSeverity,

    /// Suggested fix
    pub suggested_fix: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Critical, // Breaks proven invariant
    High,     // Breaks empirical invariant with high confidence
    Medium,   // Breaks empirical invariant with medium confidence
    Low,      // Violates heuristic
}

impl fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationSeverity::Critical => write!(f, "CRITICAL"),
            ViolationSeverity::High => write!(f, "HIGH"),
            ViolationSeverity::Medium => write!(f, "MEDIUM"),
            ViolationSeverity::Low => write!(f, "LOW"),
        }
    }
}

/// Layer information inferred from call graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Function or module name
    pub name: String,

    /// Inferred layer number (0 = leaf, higher = more dependencies)
    pub layer: usize,

    /// Names of direct dependencies (callees)
    pub dependencies: Vec<String>,

    /// Maximum layer of all dependencies
    pub max_dependency_layer: Option<usize>,
}

/// Complete invariant analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantAnalysisResult {
    /// All detected invariants
    pub invariants: Vec<Invariant>,

    /// Detected violations
    pub violations: Vec<InvariantViolation>,

    /// Layer assignments for all functions/modules
    pub layer_assignments: HashMap<String, LayerInfo>,

    /// Statistics
    pub stats: InvariantStats,
}

impl InvariantAnalysisResult {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
            violations: Vec::new(),
            layer_assignments: HashMap::new(),
            stats: InvariantStats::default(),
        }
    }

    /// Add an invariant to the result
    pub fn add_invariant(&mut self, invariant: Invariant) {
        // Update statistics
        match invariant.strength {
            InvariantStrength::Proven => self.stats.proven_count += 1,
            InvariantStrength::Empirical { .. } => self.stats.empirical_count += 1,
            InvariantStrength::Heuristic => self.stats.heuristic_count += 1,
        }

        match &invariant.kind {
            InvariantKind::Structural(_) => self.stats.structural_count += 1,
            InvariantKind::Semantic(_) => self.stats.semantic_count += 1,
            InvariantKind::Delta(_) => self.stats.delta_count += 1,
            InvariantKind::PathIntersection(_) => self.stats.path_intersection_count += 1,
        }

        self.invariants.push(invariant);
    }

    /// Add a violation to the result
    pub fn add_violation(&mut self, violation: InvariantViolation) {
        self.stats.violation_count += 1;
        self.violations.push(violation);
    }

    /// Get all blocking invariants for a specific target
    #[allow(dead_code)]
    pub fn get_blocking_invariants(&self, target: &str) -> Vec<&Invariant> {
        self.invariants
            .iter()
            .filter(|inv| inv.target == target && inv.is_blocking())
            .collect()
    }
}

impl Default for InvariantAnalysisResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about detected invariants
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InvariantStats {
    /// Total number of invariants
    pub total_count: usize,

    /// Count by strength
    pub proven_count: usize,
    pub empirical_count: usize,
    pub heuristic_count: usize,

    /// Count by kind
    pub structural_count: usize,
    pub semantic_count: usize,
    pub delta_count: usize,
    pub path_intersection_count: usize,

    /// Number of violations
    pub violation_count: usize,
}

impl InvariantStats {
    pub fn update_totals(&mut self) {
        self.total_count = self.proven_count + self.empirical_count + self.heuristic_count;
    }

    /// Get percentage of proven invariants
    pub fn proven_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.proven_count as f64 / self.total_count as f64) * 100.0
        }
    }

    /// Get percentage of heuristics
    pub fn heuristic_percentage(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.heuristic_count as f64 / self.total_count as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_from_strength() {
        assert_eq!(Confidence::from_strength(&InvariantStrength::Proven).value(), 1.0);
        assert_eq!(
            Confidence::from_strength(&InvariantStrength::Empirical { paths_checked: 150 }).value(),
            0.9
        );
        assert_eq!(Confidence::from_strength(&InvariantStrength::Heuristic).value(), 0.3);
    }

    #[test]
    fn test_is_blocking() {
        let proven_inv = Invariant::new(
            "test_fn".to_string(),
            "test.rs".to_string(),
            InvariantKind::Structural(StructuralInvariant::LayerFixed { layer: 0 }),
            InvariantStrength::Proven,
            "Test invariant".to_string(),
        );
        assert!(proven_inv.is_blocking());

        let heuristic_inv = Invariant::new(
            "test_fn".to_string(),
            "test.rs".to_string(),
            InvariantKind::Semantic(SemanticInvariant::PureFunction),
            InvariantStrength::Heuristic,
            "Test heuristic".to_string(),
        );
        assert!(!heuristic_inv.is_blocking());
    }

    #[test]
    fn test_stats_calculation() {
        let mut result = InvariantAnalysisResult::new();

        result.add_invariant(Invariant::new(
            "fn1".to_string(),
            "test.rs".to_string(),
            InvariantKind::Structural(StructuralInvariant::Leaf),
            InvariantStrength::Proven,
            "Leaf node".to_string(),
        ));

        result.add_invariant(Invariant::new(
            "fn2".to_string(),
            "test.rs".to_string(),
            InvariantKind::Semantic(SemanticInvariant::PureFunction),
            InvariantStrength::Heuristic,
            "Pure function".to_string(),
        ));

        result.stats.update_totals();

        assert_eq!(result.stats.total_count, 2);
        assert_eq!(result.stats.proven_count, 1);
        assert_eq!(result.stats.heuristic_count, 1);
        assert_eq!(result.stats.proven_percentage(), 50.0);
    }
}
