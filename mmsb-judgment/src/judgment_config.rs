//! Judgment Configuration Schemas

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentArtifact {
    pub schema: String,
    pub judgment_id: String,
    pub token: String,
    pub intent_path: String,
    pub delta_hash: String,
    pub acknowledged: bool,
    pub issued_at: String,
    pub issuer: String,
    pub version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgmentScope {
    pub schema: String,
    pub scope_id: String,
    pub judgment_id: String,
    pub intent_class: Vec<String>,
    pub allowed_paths: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub allowed_tools: Vec<String>,
    pub forbidden_tools: Vec<String>,
    pub max_files_touched: usize,
    pub max_diff_lines: usize,
    pub issued_at: String,
    pub version: u32,
}

impl JudgmentArtifact {
    pub fn new(
        token: String,
        intent_path: String,
        delta_hash: String,
    ) -> Self {
        Self {
            schema: "judgment.v1".to_string(),
            judgment_id: Uuid::new_v4().to_string(),
            token,
            intent_path,
            delta_hash,
            acknowledged: true,
            issued_at: Utc::now().to_rfc3339(),
            issuer: "human".to_string(),
            version: 1,
        }
    }
    
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    pub fn load_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&content)?)
    }
    
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl JudgmentScope {
    pub fn new(
        judgment_id: String,
        intent_class: Vec<String>,
        allowed_paths: Vec<String>,
        forbidden_paths: Vec<String>,
        allowed_tools: Vec<String>,
        forbidden_tools: Vec<String>,
        max_files_touched: usize,
        max_diff_lines: usize,
    ) -> Self {
        Self {
            schema: "judgment_scope.v1".to_string(),
            scope_id: Uuid::new_v4().to_string(),
            judgment_id,
            intent_class,
            allowed_paths,
            forbidden_paths,
            allowed_tools,
            forbidden_tools,
            max_files_touched,
            max_diff_lines,
            issued_at: Utc::now().to_rfc3339(),
            version: 1,
        }
    }
    
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    pub fn load_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&content)?)
    }
    
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}
