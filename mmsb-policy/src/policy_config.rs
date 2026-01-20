//! Policy Configuration

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub schema: String,
    pub scope_id: String,
    pub allowed_classes: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub forbidden_tools: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_files_touched: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_diff_lines: Option<usize>,
    pub version: u32,
}

impl PolicyConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    pub fn load_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&content)?)
    }
    
    pub fn matches_path(&self, path: &str) -> bool {
        for pattern in &self.forbidden_paths {
            if Self::glob_match(pattern, path) {
                return false;
            }
        }
        
        for pattern in &self.allowed_paths {
            if Self::glob_match(pattern, path) {
                return true;
            }
        }
        
        false
    }
    
    fn glob_match(pattern: &str, path: &str) -> bool {
        if pattern.contains("**") {
            let prefix = pattern.split("/**").next().unwrap_or("");
            path.starts_with(prefix)
        } else if pattern.ends_with('*') {
            let prefix = pattern.trim_end_matches('*');
            path.starts_with(prefix)
        } else {
            pattern == path
        }
    }
}
