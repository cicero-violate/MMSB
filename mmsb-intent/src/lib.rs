//! MMSB Intent Module
//!
//! Introduces new intents into the system.
//! Produces: IntentProof (A)
//! Authority: NONE

use mmsb_events::{EventSink, IntentCreated};
use mmsb_proof::{Hash, IntentBounds, IntentProof, IntentStage, ProduceProof};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
   pub description: String,
    pub intent_class: String,
    pub target_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub files_touched: usize,
    pub diff_lines: usize,
}

pub struct IntentInput {
    pub intent: Intent,
}

pub struct IntentModule<S: EventSink> {
    sink: Option<S>,
    time: u64,
}

impl<S: EventSink> IntentModule<S> {
    pub fn new() -> Self {
        Self {
            sink: None,
            time: 0,
        }
    }

    pub fn with_sink(mut self, sink: S) -> Self {
        self.sink = Some(sink);
        self
    }

   pub fn submit_intent(&mut self, intent: Intent) -> IntentProof {
        let input = IntentInput { intent: intent.clone() };
       let proof = Self::produce_proof(&input);

       self.time += 1;
       let event = IntentCreated {
           event_id: proof.intent_hash,
           timestamp: self.time,
           intent_hash: proof.intent_hash,
           intent_proof: proof.clone(),
            intent_class: intent.intent_class,
            target_paths: intent.target_paths,
            tools_used: intent.tools_used,
            files_touched: intent.files_touched,
            diff_lines: intent.diff_lines,
       };

       if let Some(sink) = &self.sink {
            sink.emit(mmsb_events::AnyEvent::IntentCreated(event));
        }

        proof
    }
}

impl<S: EventSink> Default for IntentModule<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: EventSink> ProduceProof for IntentModule<S> {
    type Input = IntentInput;
    type Proof = IntentProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let intent_hash = [0u8; 32]; // Simplified
        let bounds = IntentBounds {
            max_duration_ms: 5000,
            max_memory_bytes: 1024 * 1024,
        };
        IntentProof::new(intent_hash, 1, bounds)
    }
}

impl<S: EventSink> IntentStage for IntentModule<S> {}
