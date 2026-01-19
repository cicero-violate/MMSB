//! IntentProof (A) - Example using mmsb-proof module

use mmsb_proof::{Proof, Hash, IntentProof, IntentBounds};

fn main() {
    let intent_hash: Hash = [0u8; 32];
    let bounds = IntentBounds {
        max_duration_ms: 5000,
        max_memory_bytes: 1024 * 1024,
    };
    
    let proof = IntentProof::new(intent_hash, 1, bounds);
    
    println!("IntentProof hash: {:?}", proof.hash());
    println!("Previous: {:?}", proof.previous());
}
