//! Semantic Invariant Detection
//!
//! This module detects invariants based on function semantics, types, and effects.
//! Some are EMPIRICAL (derived from signatures), others are HEURISTIC (name-based).
//! All heuristics are clearly marked as low confidence.

use crate::invariant_types::*;
use crate::types::{CodeElement, ElementType};
use regex::Regex;

/// Semantic invariant detector
pub struct SemanticDetector<'a> {
    elements: &'a [CodeElement],
}

impl<'a> SemanticDetector<'a> {
    pub fn new(elements: &'a [CodeElement]) -> Self {
        Self { elements }
    }

    /// Detect all semantic invariants
    pub fn detect_all(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        // Detect type-stable functions
        invariants.extend(self.detect_type_stable());

        // Detect pure functions (heuristic - based on name/signature patterns)
        invariants.extend(self.detect_pure_function());

        // Detect idempotent functions (heuristic)
        invariants.extend(self.detect_idempotent());

        invariants
    }

    /// Detect type-stable functions (EMPIRICAL)
    ///
    /// Functions with explicit type signatures have stable types
    fn detect_type_stable(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        for element in self.elements {
            if !matches!(element.element_type, ElementType::Function) {
                continue;
            }

            // If signature is non-empty and looks like it has type info
            if !element.signature.is_empty() && element.signature.contains("->") {
                let inv = Invariant::new(
                    element.name.clone(),
                    element.file_path.clone(),
                    InvariantKind::Semantic(SemanticInvariant::TypeStable {
                        signature: element.signature.clone(),
                    }),
                    InvariantStrength::Empirical { paths_checked: 1 },
                    format!("Type signature: {}", element.signature),
                );

                invariants.push(inv);
            }
        }

        invariants
    }

    /// Detect pure functions (HEURISTIC)
    ///
    /// IMPORTANT: This is a heuristic based on naming patterns and signature analysis.
    /// It has LOW confidence and should not block refactorings.
    fn detect_pure_function(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        // Heuristic patterns for pure functions
        let pure_patterns = [
            r"^is_",        // is_valid, is_empty
            r"^has_",       // has_property
            r"^get_",       // get_value (might be pure)
            r"^calculate_", // calculate_sum
            r"^compute_",   // compute_hash
            r"^parse_",     // parse_input
            r"^format_",    // format_output
            r"^convert_",   // convert_type
        ];

        let mut pure_regex_set = Vec::new();
        for pattern in &pure_patterns {
            if let Ok(re) = Regex::new(pattern) {
                pure_regex_set.push(re);
            }
        }

        for element in self.elements {
            if !matches!(element.element_type, ElementType::Function) {
                continue;
            }

            let mut is_likely_pure = false;
            let mut evidence = Vec::new();

            // Check naming patterns
            for re in &pure_regex_set {
                if re.is_match(&element.name) {
                    is_likely_pure = true;
                    evidence.push(format!("Name matches pure pattern: {}", re.as_str()));
                }
            }

            // Check signature for no &mut parameters (simple heuristic)
            if !element.signature.contains("&mut") {
                evidence.push("No mutable references in signature".to_string());
            } else {
                is_likely_pure = false; // &mut suggests mutation
                evidence.clear(); // Clear evidence since this is not pure
            }

            // Check if function doesn't call many other functions (might be pure)
            // Only consider this if we haven't already ruled it out
            if element.calls.is_empty() && is_likely_pure {
                evidence.push("Calls no other functions".to_string());
            }

            if is_likely_pure {
                let mut inv = Invariant::new(
                    element.name.clone(),
                    element.file_path.clone(),
                    InvariantKind::Semantic(SemanticInvariant::PureFunction),
                    InvariantStrength::Heuristic, // Always heuristic - needs verification!
                    "Likely pure function (HEURISTIC - verify manually)".to_string(),
                );

                for e in evidence {
                    inv.add_evidence(e);
                }

                invariants.push(inv);
            }
        }

        invariants
    }

    /// Detect idempotent functions (HEURISTIC)
    ///
    /// Functions that are likely idempotent based on naming patterns
    fn detect_idempotent(&self) -> Vec<Invariant> {
        let mut invariants = Vec::new();

        let idempotent_patterns = [
            r"^set_",       // set_value (idempotent if pure set)
            r"^clear_",     // clear_buffer
            r"^reset_",     // reset_state
            r"^initialize_", // initialize_config
            r"^normalize_", // normalize_value
        ];

        let mut idempotent_regex_set = Vec::new();
        for pattern in &idempotent_patterns {
            if let Ok(re) = Regex::new(pattern) {
                idempotent_regex_set.push(re);
            }
        }

        for element in self.elements {
            if !matches!(element.element_type, ElementType::Function) {
                continue;
            }

            for re in &idempotent_regex_set {
                if re.is_match(&element.name) {
                    let mut inv = Invariant::new(
                        element.name.clone(),
                        element.file_path.clone(),
                        InvariantKind::Semantic(SemanticInvariant::Idempotent),
                        InvariantStrength::Heuristic, // Always heuristic!
                        "Likely idempotent function (HEURISTIC - verify manually)".to_string(),
                    );

                    inv.add_evidence(format!("Name matches idempotent pattern: {}", re.as_str()));

                    invariants.push(inv);
                    break; // Only add once per function
                }
            }
        }

        invariants
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Language, Visibility};

    fn make_function(name: &str, signature: &str, calls: Vec<String>) -> CodeElement {
        CodeElement {
            name: name.to_string(),
            file_path: "test.rs".to_string(),
            line_number: 1,
            element_type: ElementType::Function,
            signature: signature.to_string(),
            visibility: Visibility::Public,
            generic_params: Vec::new(),
            language: Language::Rust,
            layer: "0".to_string(),
            calls,
        }
    }

    #[test]
    fn test_detect_type_stable() {
        let elements = vec![make_function(
            "test_fn",
            "fn test_fn(x: i32) -> i32",
            Vec::new(),
        )];

        let detector = SemanticDetector::new(&elements);
        let invariants = detector.detect_type_stable();

        assert_eq!(invariants.len(), 1);
        assert_eq!(invariants[0].target, "test_fn");
        assert!(matches!(
            invariants[0].strength,
            InvariantStrength::Empirical { .. }
        ));
    }

    #[test]
    fn test_detect_pure_function_heuristic() {
        let elements = vec![
            make_function("is_valid", "fn is_valid(x: i32) -> bool", Vec::new()),
            make_function("get_value", "fn get_value() -> i32", Vec::new()),
        ];

        let detector = SemanticDetector::new(&elements);
        let invariants = detector.detect_pure_function();

        // Should detect both as potentially pure
        assert!(invariants.len() >= 2);

        // All should be heuristic
        for inv in &invariants {
            assert!(matches!(inv.strength, InvariantStrength::Heuristic));
        }
    }

    #[test]
    fn test_detect_idempotent_heuristic() {
        let elements = vec![
            make_function("set_value", "fn set_value(x: i32)", Vec::new()),
            make_function("reset_state", "fn reset_state()", Vec::new()),
        ];

        let detector = SemanticDetector::new(&elements);
        let invariants = detector.detect_idempotent();

        assert_eq!(invariants.len(), 2);

        // All should be heuristic
        for inv in &invariants {
            assert!(matches!(inv.strength, InvariantStrength::Heuristic));
            assert!(matches!(
                inv.kind,
                InvariantKind::Semantic(SemanticInvariant::Idempotent)
            ));
        }
    }

    #[test]
    fn test_no_pure_for_mutable() {
        let elements = vec![make_function(
            "is_valid",
            "fn is_valid(x: &mut i32) -> bool",
            Vec::new(),
        )];

        let detector = SemanticDetector::new(&elements);
        let invariants = detector.detect_pure_function();

        // Should not detect as pure due to &mut
        let pure_count = invariants
            .iter()
            .filter(|inv| inv.target == "is_valid")
            .count();
        assert_eq!(pure_count, 0);
    }
}
