use crate::types::ScopePolicy;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

/// Manages loading and storing scope policies
pub struct ScopeManager;

impl ScopeManager {
    /// Load a scope policy from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> io::Result<ScopePolicy> {
        let content = fs::read_to_string(path)?;
        let policy: ScopePolicy = serde_json::from_str(&content)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(policy)
    }

    /// Save a scope policy to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(path: P, policy: &ScopePolicy) -> io::Result<()> {
        let content = serde_json::to_string_pretty(policy)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Load the default scope policy
    pub fn load_default() -> ScopePolicy {
        ScopePolicy::default_permissive()
    }
}

/// JSON schema representation for scope policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopePolicyV1 {
    pub schema: String,
    pub scope_id: String,
    pub allowed_classes: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub forbidden_tools: Vec<String>,
    pub max_files_touched: Option<usize>,
    pub max_diff_lines: Option<usize>,
    pub version: u32,
}

impl From<ScopePolicy> for ScopePolicyV1 {
    fn from(policy: ScopePolicy) -> Self {
        Self {
            schema: "intent_policy.v1".to_string(),
            scope_id: policy.scope_id,
            allowed_classes: policy
                .allowed_classes
                .iter()
                .map(|c| c.as_str().to_string())
                .collect(),
            allowed_paths: policy.allowed_paths,
            forbidden_paths: policy.forbidden_paths,
            allowed_tools: policy.allowed_tools,
            forbidden_tools: policy.forbidden_tools,
            max_files_touched: policy.max_files_touched,
            max_diff_lines: policy.max_diff_lines,
            version: 1,
        }
    }
}

impl From<ScopePolicyV1> for ScopePolicy {
    fn from(v1: ScopePolicyV1) -> Self {
        let mut policy = ScopePolicy::new(v1.scope_id);
        for class in v1.allowed_classes {
            policy.allowed_classes.insert(class.into());
        }
        policy.allowed_paths = v1.allowed_paths;
        policy.forbidden_paths = v1.forbidden_paths;
        policy.allowed_tools = v1.allowed_tools;
        policy.forbidden_tools = v1.forbidden_tools;
        policy.max_files_touched = v1.max_files_touched;
        policy.max_diff_lines = v1.max_diff_lines;
        policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IntentClass;

    #[test]
    fn test_policy_serialization() {
        let mut policy = ScopePolicy::new("test_scope");
        policy.allowed_classes.insert(IntentClass::formatting());
        policy.allowed_paths.push("src/".to_string());
        policy.forbidden_paths.push("migrations/".to_string());
        policy.max_files_touched = Some(50);

        let v1: ScopePolicyV1 = policy.clone().into();
        let roundtrip: ScopePolicy = v1.into();

        assert_eq!(roundtrip.scope_id, policy.scope_id);
        assert_eq!(roundtrip.allowed_paths, policy.allowed_paths);
        assert_eq!(roundtrip.forbidden_paths, policy.forbidden_paths);
        assert_eq!(roundtrip.max_files_touched, policy.max_files_touched);
    }
}
