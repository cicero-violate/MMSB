//! MMSB Intent Module
//!
//! Introduces new intents into the system.
//! Produces: IntentProof (A)
//! Authority: NONE

use mmsb_proof::{Hash, IntentBounds, IntentCreated, IntentProof, IntentStage, ProduceProof};
use mmsb_service::{EventBus, Module};
use serde::{Deserialize, Serialize};

/// Raw intent input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawIntent {
    pub content: String,
    pub max_duration_ms: u64,
    pub max_memory_bytes: u64,
}

impl RawIntent {
    pub fn hash(&self) -> Hash {
        // Simplified hash - in production use proper crypto hash
        let mut hash = [0u8; 32];
        let bytes = self.content.as_bytes();
        for (i, byte) in bytes.iter().take(32).enumerate() {
            hash[i] = *byte;
        }
        hash
    }
}

/// Intent module implementation
pub struct IntentModule {
    event_bus: Option<EventBus>,
    logical_time: u64,
}

impl IntentModule {
    pub fn new() -> Self {
        Self {
            event_bus: None,
            logical_time: 0,
        }
    }
    
    fn next_time(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }
}

impl Default for IntentModule {
    fn default() -> Self {
        Self::new()
    }
}

impl ProduceProof for IntentModule {
    type Input = RawIntent;
    type Proof = IntentProof;
    
    fn produce_proof(input: &Self::Input) -> Self::Proof {
        let bounds = IntentBounds {
            max_duration_ms: input.max_duration_ms,
            max_memory_bytes: input.max_memory_bytes,
        };
        
        IntentProof::new(input.hash(), 1, bounds)
    }
}

impl IntentStage for IntentModule {}

impl IntentModule {
    pub fn create_intent(&mut self, raw_intent: RawIntent) {
        let proof = Self::produce_proof(&raw_intent);
        let intent_hash = raw_intent.hash();
        
        let event = IntentCreated {
            event_id: intent_hash,
            timestamp: self.next_time(),
            intent_hash,
            intent_proof: proof,
        };
        
        if let Some(bus) = &self.event_bus {
            let _ = bus.emit(mmsb_proof::AnyEvent::IntentCreated(event));
        }
    }
}

impl Module for IntentModule {
    fn name(&self) -> &str {
        "mmsb-intent"
    }
    
    fn initialize(&mut self, bus: EventBus) {
        self.event_bus = Some(bus);
    }
    
    fn shutdown(&mut self) {
        self.event_bus = None;
    }
}
