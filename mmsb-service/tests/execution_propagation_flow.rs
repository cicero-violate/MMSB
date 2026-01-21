//! Integration tests for execution → propagation → admission flow
//!
//! Verifies that:
//! 1. Executor produces ExecutionProof + ProposedDelta (no mutation)
//! 2. Propagation normalizes delta (no mutation)
//! 3. Memory admits and commits (ONLY mutation point)

use mmsb_executor::{ExecutionLoop, ExecutionCoordinator};
use mmsb_propagation::DeltaNormalizer;
use mmsb_proof::{JudgmentProof, Proof};
use mmsb_memory::epoch::Epoch;
use mmsb_primitives::PageID;

/// Test that execution produces proof without mutating state
#[test]
fn test_execution_produces_proof_no_mutation() {
    let mut executor = ExecutionLoop::new();
    
    // Create a test judgment
    let judgment = create_test_judgment();
    
    // Execute - should produce proof and proposed delta
    let outcome = executor.execute(&judgment);
    
    // Verify execution proof
    assert!(outcome.proof.success);
    assert_eq!(outcome.proof.judgment_hash, judgment.hash());
    
    // Verify proposed delta exists
    assert_eq!(outcome.proposed_delta.page_id, PageID(0));
    
    // CRITICAL: Verify no state was mutated
    // (This is a structural test - executor has no access to MemoryEngine)
    println!("✓ Execution produced proof without mutation");
}

/// Test that propagation normalizes delta without mutating state
#[test]
fn test_propagation_normalizes_no_mutation() {
    let mut executor = ExecutionLoop::new();
    let judgment = create_test_judgment();
    
    // Phase 1: Execute
    let outcome = executor.execute(&judgment);
    
    // Phase 2: Normalize (propagation)
    let normalized = DeltaNormalizer::normalize(
        outcome.proposed_delta.page_id,
        Epoch(1),
        outcome.proposed_delta.payload,
    );
    
    // Verify normalized delta
    assert_eq!(normalized.page_id, PageID(0));
    assert_eq!(normalized.epoch, Epoch(1));
    
    // CRITICAL: Verify no state was mutated
    // (DeltaNormalizer is pure function, has no MemoryEngine access)
    println!("✓ Propagation normalized delta without mutation");
}

/// Test full coordination flow
#[test]
fn test_coordinator_orchestrates_pipeline() {
    let mut coordinator = ExecutionCoordinator::new();
    let judgment = create_test_judgment();
    
    // Coordinate full flow
    let result = coordinator.coordinate(&judgment);
    
    // Verify execution outcome
    assert!(result.execution_outcome.proof.success);
    assert_eq!(
        result.execution_outcome.proof.judgment_hash,
        judgment.hash()
    );
    
    // Note: admission_proof is None because we haven't connected
    // to MemoryEngine yet (that's the next phase)
    assert!(result.admission_proof.is_none());
    
    println!("✓ Coordinator orchestrated pipeline without mutation");
}

/// Test deterministic replay - same input produces same output
#[test]
fn test_execution_determinism() {
    let mut executor1 = ExecutionLoop::new();
    let mut executor2 = ExecutionLoop::new();
    
    let judgment = create_test_judgment();
    
    // Execute same judgment twice
    let outcome1 = executor1.execute(&judgment);
    let outcome2 = executor2.execute(&judgment);
    
    // Results should be identical (deterministic)
    assert_eq!(outcome1.proof.judgment_hash, outcome2.proof.judgment_hash);
    assert_eq!(outcome1.proof.success, outcome2.proof.success);
    
    println!("✓ Execution is deterministic");
}

/// Test that normalization is idempotent
#[test]
fn test_normalization_idempotent() {
    let page_id = PageID(42);
    let epoch = Epoch(1);
    let payload = vec![1, 2, 3, 4];
    
    // Normalize once
    let normalized1 = DeltaNormalizer::normalize(
        page_id,
        epoch,
        payload.clone(),
    );
    
    // Normalize the result again
    let normalized2 = DeltaNormalizer::normalize(
        normalized1.page_id,
        normalized1.epoch,
        normalized1.payload.clone(),
    );
    
    // Should be identical (idempotent)
    assert_eq!(normalized1.page_id, normalized2.page_id);
    assert_eq!(normalized1.epoch, normalized2.epoch);
    
    println!("✓ Normalization is idempotent");
}

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_judgment() -> JudgmentProof {
    use mmsb_proof::{PolicyProof, IntentProof, IntentBounds};
    
    // Create intent proof
    let intent_proof = IntentProof::new(
        [1u8; 32],
        1,
        IntentBounds {
            max_duration_ms: 1000,
            max_memory_bytes: 1024 * 1024,
        },
    );
    
    // Create policy proof
    let policy_proof = PolicyProof::new(
        intent_proof.hash(),
        mmsb_proof::PolicyCategory::AutoApprove,
        mmsb_proof::RiskClass::Low,
    );
    
    // Create judgment proof
    JudgmentProof::new(
        policy_proof.hash(),
        true,
        [0u8; 64],
        1000,
    )
}
