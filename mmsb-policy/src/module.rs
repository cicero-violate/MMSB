//! MMSB Policy Module

use mmsb_events::{IntentCreated, PolicyEvaluated};
use mmsb_proof::{IntentProof, PolicyCategory, PolicyProof, PolicyStage, Proof, ProduceProof, RiskClass};
use crate::policy_config::PolicyConfig;

pub struct PolicyInput {
    pub intent_proof: IntentProof,
    pub intent_class: String,
    pub target_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub files_touched: usize,
    pub diff_lines: usize,
}

pub struct PolicyModule {
    logical_time: u64,
    config: PolicyConfig,
}

impl PolicyModule {
    pub fn new() -> Self {
        Self::with_config(PolicyConfig {
            schema: "intent_policy.v1".to_string(),
            scope_id: "default".to_string(),
            allowed_classes: vec![
                "formatting".to_string(),
                "lint_fix".to_string(),
                "documentation".to_string(),
            ],
            allowed_paths: vec!["src/**/*.rs".to_string()],
            forbidden_paths: vec![
                "migrations/**".to_string(),
                ".git/**".to_string(),
            ],
            allowed_tools: vec!["rustfmt".to_string(), "clippy".to_string()],
            forbidden_tools: vec![
                "shell_runner".to_string(),
                "arbitrary_exec".to_string(),
            ],
            max_files_touched: Some(50),
            max_diff_lines: Some(2000),
            version: 1,
        })
    }
    
    pub fn with_config(config: PolicyConfig) -> Self {
        Self {
            logical_time: 0,
            config,
        }
    }
    
    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = PolicyConfig::load_file(path)?;
        Ok(Self::with_config(config))
    }

    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

    fn classify_risk(&self, input: &PolicyInput) -> RiskClass {
        // Check forbidden tools
        for tool in &input.tools_used {
            if self.config.forbidden_tools.contains(tool) {
                return RiskClass::Critical;
            }
        }
        
        // Check forbidden paths
        for path in &input.target_paths {
            if !self.config.matches_path(path) {
                return RiskClass::High;
            }
        }

        // Check scope limits
        if let Some(max_files) = self.config.max_files_touched {
            if input.files_touched > max_files {
                return RiskClass::High;
            }
        }
        
        if let Some(max_lines) = self.config.max_diff_lines {
            if input.diff_lines > max_lines {
                return RiskClass::High;
            }
        }

        // Check allowed classes
        if self.config.allowed_classes.contains(&input.intent_class) {
            RiskClass::Low
        } else {
            RiskClass::Medium
        }
    }

    fn determine_category(risk: RiskClass) -> PolicyCategory {
        match risk {
            RiskClass::Low => PolicyCategory::AutoApprove,
            RiskClass::Medium => PolicyCategory::RequiresReview,
            RiskClass::High | RiskClass::Critical => PolicyCategory::RequiresReview,
        }
    }

    pub fn handle_intent_created(&mut self, event: IntentCreated) -> PolicyEvaluated {
        let input = PolicyInput {
            intent_proof: event.intent_proof.clone(),
            intent_class: event.intent_class,
            target_paths: event.target_paths,
            tools_used: event.tools_used,
            files_touched: event.files_touched,
            diff_lines: event.diff_lines,
        };

        let proof = Self::produce_proof(&input);

        PolicyEvaluated {
            event_id: event.intent_hash,
            timestamp: self.next_time(),
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof,
            policy_proof: proof,
        }
    }
}

impl Default for PolicyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ProduceProof for PolicyModule {
    type Input = PolicyInput;
    type Proof = PolicyProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let module = PolicyModule::new();
        let risk_class = module.classify_risk(input);
        let category = Self::determine_category(risk_class.clone());
        PolicyProof::new(input.intent_proof.hash(), category, risk_class)
    }
}

impl PolicyStage for PolicyModule {}
