//! MMSB Policy Module - implements PolicyStage trait

use mmsb_events::{EventSink, IntentCreated, PolicyEvaluated};
use mmsb_proof::{IntentProof, PolicyCategory, PolicyProof, PolicyStage, Proof, ProduceProof, RiskClass};

pub struct PolicyInput {
    pub intent_proof: IntentProof,
    pub intent_class: String,
    pub target_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub files_touched: usize,
    pub diff_lines: usize,
}

pub struct PolicyModule<S: EventSink> {
    sink: Option<S>,
    logical_time: u64,
    allowed_classes: Vec<String>,
    forbidden_tools: Vec<String>,
    max_files_touched: usize,
    max_diff_lines: usize,
}

impl<S: EventSink> PolicyModule<S> {
    pub fn new() -> Self {
        Self {
            sink: None,
            logical_time: 0,
            allowed_classes: vec![
                "formatting".to_string(),
                "lint_fix".to_string(),
                "documentation".to_string(),
            ],
            forbidden_tools: vec![
                "shell_runner".to_string(),
                "arbitrary_exec".to_string(),
            ],
            max_files_touched: 50,
            max_diff_lines: 2000,
        }
    }

    pub fn with_sink(mut self, sink: S) -> Self {
        self.sink = Some(sink);
        self
    }

    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

    fn classify_risk(&self, input: &PolicyInput) -> RiskClass {
        for tool in &input.tools_used {
            if self.forbidden_tools.contains(tool) {
                return RiskClass::Critical;
            }
        }

        if input.files_touched > self.max_files_touched || input.diff_lines > self.max_diff_lines {
            return RiskClass::High;
        }

        if self.allowed_classes.contains(&input.intent_class) {
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
}

impl<S: EventSink> Default for PolicyModule<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: EventSink> ProduceProof for PolicyModule<S> {
    type Input = PolicyInput;
    type Proof = PolicyProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let module = PolicyModule::<S>::new();
        let risk_class = module.classify_risk(input);
        let category = Self::determine_category(risk_class.clone());
        PolicyProof::new(input.intent_proof.hash(), category, risk_class)
    }
}

impl<S: EventSink> PolicyStage for PolicyModule<S> {}

impl<S: EventSink> PolicyModule<S> {
    pub fn handle_intent_created(&mut self, event: IntentCreated) {
        let input = PolicyInput {
            intent_proof: event.intent_proof.clone(),
            intent_class: event.intent_class,
            target_paths: event.target_paths,
            tools_used: event.tools_used,
            files_touched: event.files_touched,
            diff_lines: event.diff_lines,
        };

        let proof = Self::produce_proof(&input);

        let policy_event = PolicyEvaluated {
            event_id: event.intent_hash,
            timestamp: self.next_time(),
            intent_hash: event.intent_hash,
            intent_proof: event.intent_proof,
            policy_proof: proof,
        };

        if let Some(sink) = &self.sink {
            sink.emit(policy_event);
        }
    }
}
