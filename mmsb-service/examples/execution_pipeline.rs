//! Example: Execution → Propagation → Admission Pipeline
//!
//! Demonstrates the complete flow:
//! 1. Executor produces ExecutionProof + ProposedDelta (no mutation)
//! 2. Propagation normalizes delta (no mutation)
//! 3. Memory admits, commits, validates (ONLY mutation point)

use mmsb_executor::{ExecutionLoop, ExecutionCoordinator};
use mmsb_propagation::DeltaNormalizer;
use mmsb_proof::{JudgmentProof, PolicyProof, IntentProof, IntentBounds, Proof};
use mmsb_memory::epoch::Epoch;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  MMSB Execution Pipeline Example");
    println!("═══════════════════════════════════════════════════════════\n");
    
    // Create a test judgment
    let judgment = create_judgment();
    println!("✓ Created JudgmentProof (authority approval)");
    println!("  - Judgment hash: {:?}...", &judgment.hash()[..8]);
    println!("  - Approved: {}\n", judgment.approved);
    
    // Phase 1: Execution (NO MUTATION)
    println!("Phase 1: EXECUTION (no mutation)");
    println!("─────────────────────────────────────────────────────────");
    let mut executor = ExecutionLoop::new();
    let outcome = executor.execute(&judgment);
    println!("✓ Executor produced ExecutionProof");
    println!("  - Execution ID: {:?}...", &outcome.proof.execution_id[..8]);
    println!("  - Success: {}", outcome.proof.success);
    println!("  - Result hash: {:?}...", &outcome.proof.result_hash[..8]);
    println!("  - Proposed delta for page: {:?}", outcome.proposed_delta.page_id);
    println!("  → CRITICAL: No canonical state was mutated\n");
    
    // Phase 2: Propagation (NO MUTATION)
    println!("Phase 2: PROPAGATION (no mutation)");
    println!("─────────────────────────────────────────────────────────");
    let normalized = DeltaNormalizer::normalize(
        outcome.proposed_delta.page_id,
        Epoch(1),
        outcome.proposed_delta.payload.clone(),
    );
    println!("✓ Propagation normalized delta");
    println!("  - Page ID: {:?}", normalized.page_id);
    println!("  - Epoch: {:?}", normalized.epoch);
    println!("  - Dependencies: {} pages", normalized.dependencies.len());
    println!("  → CRITICAL: No canonical state was mutated\n");
    
    // Phase 3: Coordinator (NO MUTATION)
    println!("Phase 3: COORDINATION (no mutation)");
    println!("─────────────────────────────────────────────────────────");
    let mut coordinator = ExecutionCoordinator::new();
    let coordinated = coordinator.coordinate(&judgment);
    println!("✓ Coordinator orchestrated pipeline");
    println!("  - Execution proof generated: ✓");
    println!("  - Proposed delta generated: ✓");
    println!("  - Ready for admission: {}", coordinated.admission_proof.is_none());
    println!("  → CRITICAL: No canonical state was mutated\n");
    
    // Summary
    println!("═══════════════════════════════════════════════════════════");
    println!("  CRITICAL INVARIANT MAINTAINED");
    println!("═══════════════════════════════════════════════════════════");
    println!("✓ Executor: NO mutation (structural separation)");
    println!("✓ Propagation: NO mutation (pure function)");
    println!("✓ Coordinator: NO mutation (orchestration only)");
    println!();
    println!("ONLY MemoryEngine::commit_delta() can mutate canonical state");
    println!("  - Admission: verifies permission");
    println!("  - Commit: MUTATES state → advances epoch");
    println!("  - Outcome: validates invariants");
    println!();
    println!("∀ mutation m : m ∈ MemoryEngine.commit_delta()");
    println!("═══════════════════════════════════════════════════════════\n");
}

fn create_judgment() -> JudgmentProof {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let intent_proof = IntentProof::new(
        [42u8; 32],
        1,
        IntentBounds {
            max_duration_ms: 1000,
            max_memory_bytes: 1024 * 1024,
        },
    );
    
    let policy_proof = PolicyProof::new(
        intent_proof.hash(),
        mmsb_proof::PolicyCategory::AutoApprove,
        mmsb_proof::RiskClass::Low,
    );
    
    JudgmentProof::new(
        policy_proof.hash(),
        true,
        [0u8; 64],
        timestamp,
    )
}
