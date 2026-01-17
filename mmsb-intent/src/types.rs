use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Represents a class of intent operation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentClass(String);

impl IntentClass {
    pub fn new(class: impl Into<String>) -> Self {
        Self(class.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Common intent classes
    pub fn rust_refactor() -> Self {
        Self("rust_refactor".to_string())
    }

    pub fn formatting() -> Self {
        Self("formatting".to_string())
    }

    pub fn lint_fix() -> Self {
        Self("lint_fix".to_string())
    }

    pub fn documentation() -> Self {
        Self("documentation".to_string())
    }

    pub fn structural_change() -> Self {
        Self("structural_change".to_string())
    }

    pub fn state_change() -> Self {
        Self("state_change".to_string())
    }

    pub fn unknown() -> Self {
        Self("unknown".to_string())
    }
}

impl From<String> for IntentClass {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for IntentClass {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Metadata about an intent operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMetadata {
    /// Classification of the intent
    pub classes: Vec<IntentClass>,
    
    /// Paths that will be affected
    pub affected_paths: Vec<String>,
    
    /// Tools that will be used
    pub tools_used: Vec<String>,
    
    /// Estimated number of files touched
    pub files_touched: usize,
    
    /// Estimated number of diff lines
    pub diff_lines: usize,
}

impl IntentMetadata {
    pub fn new() -> Self {
        Self {
            classes: vec![IntentClass::unknown()],
            affected_paths: Vec::new(),
            tools_used: Vec::new(),
            files_touched: 0,
            diff_lines: 0,
        }
    }

    pub fn with_class(mut self, class: IntentClass) -> Self {
        if self.classes.len() == 1 && self.classes[0] == IntentClass::unknown() {
            self.classes.clear();
        }
        self.classes.push(class);
        self
    }

    pub fn with_classes(mut self, classes: Vec<IntentClass>) -> Self {
        self.classes = classes;
        self
    }

    pub fn with_paths(mut self, paths: Vec<String>) -> Self {
        self.affected_paths = paths;
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools_used = tools;
        self
    }

    pub fn with_files_touched(mut self, count: usize) -> Self {
        self.files_touched = count;
        self
    }

    pub fn with_diff_lines(mut self, count: usize) -> Self {
        self.diff_lines = count;
        self
    }
}

impl Default for IntentMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// A scope policy that defines what operations require judgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopePolicy {
    /// Scope identifier
    pub scope_id: String,
    
    /// Intent classes allowed to bypass judgment
    pub allowed_classes: HashSet<IntentClass>,
    
    /// Path patterns that are allowed
    pub allowed_paths: Vec<String>,
    
    /// Path patterns that are forbidden
    pub forbidden_paths: Vec<String>,
    
    /// Tools that are allowed
    pub allowed_tools: Vec<String>,
    
    /// Tools that are forbidden
    pub forbidden_tools: Vec<String>,
    
    /// Maximum files that can be touched
    pub max_files_touched: Option<usize>,
    
    /// Maximum diff lines allowed
    pub max_diff_lines: Option<usize>,
}

impl ScopePolicy {
    pub fn new(scope_id: impl Into<String>) -> Self {
        Self {
            scope_id: scope_id.into(),
            allowed_classes: HashSet::new(),
            allowed_paths: Vec::new(),
            forbidden_paths: Vec::new(),
            allowed_tools: Vec::new(),
            forbidden_tools: Vec::new(),
            max_files_touched: None,
            max_diff_lines: None,
        }
    }

    pub fn default_permissive() -> Self {
        let mut policy = Self::new("default_permissive");
        policy.allowed_classes.insert(IntentClass::formatting());
        policy.allowed_classes.insert(IntentClass::lint_fix());
        policy.allowed_classes.insert(IntentClass::documentation());
        policy
    }
}

/// Result of policy evaluation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// Operation can proceed without judgment
    Allow,
    
    /// Operation requires judgment
    RequireJudgment(Vec<PolicyViolation>),
}

/// Reasons why an operation requires judgment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyViolation {
    ClassNotAllowed(String),
    PathNotAllowed(String),
    PathForbidden(String),
    ToolNotAllowed(String),
    ToolForbidden(String),
    TooManyFiles { actual: usize, max: usize },
    TooManyDiffLines { actual: usize, max: usize },
    NoClassification,
}

impl PolicyViolation {
    pub fn description(&self) -> String {
        match self {
            Self::ClassNotAllowed(class) => {
                format!("Intent class '{}' not in allowed set", class)
            }
            Self::PathNotAllowed(path) => {
                format!("Path '{}' not in allowed patterns", path)
            }
            Self::PathForbidden(path) => {
                format!("Path '{}' matches forbidden pattern", path)
            }
            Self::ToolNotAllowed(tool) => {
                format!("Tool '{}' not in allowed set", tool)
            }
            Self::ToolForbidden(tool) => {
                format!("Tool '{}' is forbidden", tool)
            }
            Self::TooManyFiles { actual, max } => {
                format!("Too many files ({} > {})", actual, max)
            }
            Self::TooManyDiffLines { actual, max } => {
                format!("Too many diff lines ({} > {})", actual, max)
            }
            Self::NoClassification => {
                "Intent has no classification".to_string()
            }
        }
    }
}
