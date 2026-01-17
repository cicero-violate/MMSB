use crate::types::{IntentMetadata, PolicyDecision, PolicyViolation, ScopePolicy};
use glob::Pattern;

/// Evaluates whether an intent requires judgment based on policy
pub struct PolicyEvaluator {
    policy: ScopePolicy,
}

impl PolicyEvaluator {
    pub fn new(policy: ScopePolicy) -> Self {
        Self { policy }
    }

    /// Evaluate if the intent requires judgment
    pub fn evaluate(&self, metadata: &IntentMetadata) -> PolicyDecision {
        let mut violations = Vec::new();

        // Check if intent has classification
        if metadata.classes.is_empty() {
            violations.push(PolicyViolation::NoClassification);
            return PolicyDecision::RequireJudgment(violations);
        }

        // Check if any class is allowed
        let has_allowed_class = metadata
            .classes
            .iter()
            .any(|c| self.policy.allowed_classes.contains(c));

        if !has_allowed_class && !self.policy.allowed_classes.is_empty() {
            for class in &metadata.classes {
                violations.push(PolicyViolation::ClassNotAllowed(class.as_str().to_string()));
            }
        }

        // Check paths
        for path in &metadata.affected_paths {
            if !self.is_path_allowed(path) {
                violations.push(PolicyViolation::PathNotAllowed(path.clone()));
            }
            if self.is_path_forbidden(path) {
                violations.push(PolicyViolation::PathForbidden(path.clone()));
            }
        }

        // Check tools
        for tool in &metadata.tools_used {
            if !self.policy.allowed_tools.is_empty()
                && !self.policy.allowed_tools.contains(tool)
            {
                violations.push(PolicyViolation::ToolNotAllowed(tool.clone()));
            }
            if self.policy.forbidden_tools.contains(tool) {
                violations.push(PolicyViolation::ToolForbidden(tool.clone()));
            }
        }

        // Check file count limit
        if let Some(max) = self.policy.max_files_touched {
            if metadata.files_touched > max {
                violations.push(PolicyViolation::TooManyFiles {
                    actual: metadata.files_touched,
                    max,
                });
            }
        }

        // Check diff lines limit
        if let Some(max) = self.policy.max_diff_lines {
            if metadata.diff_lines > max {
                violations.push(PolicyViolation::TooManyDiffLines {
                    actual: metadata.diff_lines,
                    max,
                });
            }
        }

        if violations.is_empty() {
            PolicyDecision::Allow
        } else {
            PolicyDecision::RequireJudgment(violations)
        }
    }

    fn is_path_allowed(&self, path: &str) -> bool {
        if self.policy.allowed_paths.is_empty() {
            return true; // No restrictions means everything is allowed
        }

        self.policy.allowed_paths.iter().any(|pattern| {
            Self::matches_pattern(pattern, path)
        })
    }

    fn is_path_forbidden(&self, path: &str) -> bool {
        self.policy.forbidden_paths.iter().any(|pattern| {
            Self::matches_pattern(pattern, path)
        })
    }

    fn matches_pattern(pattern: &str, path: &str) -> bool {
        // If pattern ends with /, treat as prefix match (directory)
        if pattern.ends_with('/') {
            return path.starts_with(pattern);
        }
        
        // If pattern contains glob characters, use glob matching
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            if let Ok(glob) = Pattern::new(pattern) {
                return glob.matches(path);
            }
        }
        
        // Otherwise, use prefix matching
        path.starts_with(pattern)
    }

    pub fn policy(&self) -> &ScopePolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IntentClass;

    #[test]
    fn test_allow_simple_formatting() {
        let mut policy = ScopePolicy::new("test");
        policy.allowed_classes.insert(IntentClass::formatting());

        let evaluator = PolicyEvaluator::new(policy);

        let metadata = IntentMetadata::new()
            .with_class(IntentClass::formatting())
            .with_paths(vec!["src/main.rs".to_string()])
            .with_files_touched(1)
            .with_diff_lines(10);

        assert_eq!(evaluator.evaluate(&metadata), PolicyDecision::Allow);
    }

    #[test]
    fn test_reject_disallowed_class() {
        let mut policy = ScopePolicy::new("test");
        policy.allowed_classes.insert(IntentClass::formatting());

        let evaluator = PolicyEvaluator::new(policy);

        let metadata = IntentMetadata::new()
            .with_class(IntentClass::structural_change())
            .with_paths(vec!["src/main.rs".to_string()]);

        match evaluator.evaluate(&metadata) {
            PolicyDecision::RequireJudgment(violations) => {
                assert!(!violations.is_empty());
            }
            PolicyDecision::Allow => panic!("Expected rejection"),
        }
    }

    #[test]
    fn test_reject_forbidden_path() {
        let mut policy = ScopePolicy::new("test");
        policy.allowed_classes.insert(IntentClass::formatting());
        policy.forbidden_paths.push("migrations/".to_string());

        let evaluator = PolicyEvaluator::new(policy);

        let metadata = IntentMetadata::new()
            .with_class(IntentClass::formatting())
            .with_paths(vec!["migrations/001_init.sql".to_string()]);

        match evaluator.evaluate(&metadata) {
            PolicyDecision::RequireJudgment(violations) => {
                assert!(violations
                    .iter()
                    .any(|v| matches!(v, PolicyViolation::PathForbidden(_))));
            }
            PolicyDecision::Allow => panic!("Expected rejection"),
        }
    }

    #[test]
    fn test_reject_too_many_files() {
        let mut policy = ScopePolicy::new("test");
        policy.allowed_classes.insert(IntentClass::formatting());
        policy.max_files_touched = Some(5);

        let evaluator = PolicyEvaluator::new(policy);

        let metadata = IntentMetadata::new()
            .with_class(IntentClass::formatting())
            .with_files_touched(10);

        match evaluator.evaluate(&metadata) {
            PolicyDecision::RequireJudgment(violations) => {
                assert!(violations
                    .iter()
                    .any(|v| matches!(v, PolicyViolation::TooManyFiles { .. })));
            }
            PolicyDecision::Allow => panic!("Expected rejection"),
        }
    }
}
