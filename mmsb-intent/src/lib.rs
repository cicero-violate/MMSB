//! MMSB Intent Module

use mmsb_events::{IntentCreated, Intent};
use mmsb_proof::{IntentBounds, IntentProof, IntentStage, ProduceProof};
use serde::{Deserialize, Serialize};

pub struct IntentInput {
    pub intent: Intent,
}

pub struct IntentModule {
    time: u64,
}

impl IntentModule {
    pub fn new() -> Self {
        Self { time: 0 }
    }

    pub fn submit_intent(&mut self, intent: Intent) -> IntentCreated {
        let input = IntentInput { intent: intent.clone() };
        let proof = Self::produce_proof(&input);

        self.time += 1;
        IntentCreated {
            event_id: proof.intent_hash,
            timestamp: self.time,
            intent_hash: proof.intent_hash,
            intent_proof: proof,
            intent_class: intent.intent_class,
            target_paths: intent.target_paths,
            tools_used: intent.tools_used,
            files_touched: intent.files_touched,
            diff_lines: intent.diff_lines,
        }
    }
}

impl Default for IntentModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ProduceProof for IntentModule {
    type Input = IntentInput;
    type Proof = IntentProof;

    fn produce_proof(_input: &Self::Input) -> Self::Proof {
        let intent_hash = [0u8; 32];
        let bounds = IntentBounds {
            max_duration_ms: _input.intent.max_duration_ms,
            max_memory_bytes: _input.intent.max_memory_bytes,
        };
        IntentProof::new(intent_hash, 1, bounds)
    }
}

impl IntentStage for IntentModule {}
