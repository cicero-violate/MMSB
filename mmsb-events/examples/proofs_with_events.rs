//! Example: Proof chain with events

use mmsb_proof::{Proof, Hash, IntentProof, IntentBounds, PolicyProof, PolicyCategory, RiskClass};
use mmsb_events::{Event, IntentCreated, PolicyEvaluated};

fn main() {
    // 1. Create IntentProof (A)
    let intent_hash: Hash = [1u8; 32];
    let bounds = IntentBounds {
        max_duration_ms: 5000,
        max_memory_bytes: 1024 * 1024,
    };
    let intent_proof = IntentProof::new(intent_hash, 1, bounds);
    
    println!("A - IntentProof created");
    
    // 2. Emit IntentCreated event
    let intent_event = IntentCreated {
        event_id: [2u8; 32],
        timestamp: 100,
        intent_hash,
        intent_proof: intent_proof.clone(),
    };
    
    println!("Event: IntentCreated");
    
    // 3. Create PolicyProof (B) - links to A
    let policy_proof = PolicyProof::new(
        intent_proof.hash(),
        PolicyCategory::AutoApprove,
        RiskClass::Low,
    );
    
    println!("B - PolicyProof created, links to A: {:?}", policy_proof.previous().is_some());
    
    // 4. Emit PolicyEvaluated event
    let policy_event = PolicyEvaluated {
        event_id: [3u8; 32],
        timestamp: 101,
        intent_hash,
        intent_proof,
        policy_proof: policy_proof.clone(),
    };
    
    println!("Event: PolicyEvaluated");
    
    // Verify chain
    println!("\n=== Chain Verification ===");
    println!("B.previous() == A.hash(): {}", 
        policy_event.policy_proof.previous() == Some(intent_event.intent_proof.hash()));
}
