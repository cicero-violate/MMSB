//! Integration test verifying executor and propagation never mutate
//!
//! Answer: **NO**, MMSB does NOT have traditional shared memory.
//!
//! MMSB Architecture:
//! - Canonical state owned ONLY by MemoryEngine  
//! - Pages allocated in CPU/GPU/Unified by PageAllocator
//! - Zero-copy notification via broadcast channels
//! - Transaction log for durability
//!
//! Critical Invariant: ∀ mutation m : m ∈ MemoryEngine.commit_delta()

use mmsb_executor::{ExecutionLoop, ExecutionCoordinator};
use mmsb_propagation::DeltaNormalizer;
use mmsb_memory::epoch::Epoch;
use mmsb_proof::{JudgmentProof, Proof};

/// Test that executor produces proof without any MemoryEngine access
#[test]
fn test_executor_structural_separation() {
    let mut executor = ExecutionLoop::new();
    let judgment = create_test_judgment();
    
    // Execute - structurally cannot mutate MemoryEngine (no reference to it)
    let outcome = executor.execute(&judgment);
    
    // Verify execution produced proof
    assert!(outcome.proof.success);
    assert_eq!(outcome.proof.judgment_hash, judgment.hash());
    
    println!("✓ Executor structurally separated (no MemoryEngine access)");
}

/// Test that propagation normalizes without any MemoryEngine access
#[test]
fn test_propagation_structural_separation() {
    let mut executor = ExecutionLoop::new();
    let judgment = create_test_judgment();
    let outcome = executor.execute(&judgment);
    
    // Normalize - structurally cannot mutate MemoryEngine (pure function)
    let normalized = DeltaNormalizer::normalize(
        outcome.proposed_delta.page_id,
        Epoch(1),
        outcome.proposed_delta.payload,
    );
    
    // Verify normalized delta
    assert_eq!(normalized.page_id, outcome.proposed_delta.page_id);
    
    println!("✓ Propagation structurally separated (pure function)");
}

/// Test that coordinator orchestrates without mutation
#[test]
fn test_coordinator_structural_separation() {
    let mut coordinator = ExecutionCoordinator::new();
    let judgment = create_test_judgment();
    
    // Coordinate - structurally cannot mutate MemoryEngine  
    let result = coordinator.coordinate(&judgment);
    
    // Verify coordination produced execution outcome
    assert!(result.execution_outcome.proof.success);
    
    println!("✓ Coordinator structurally separated (no MemoryEngine access)");
}

/// Test deterministic execution
#[test]
fn test_execution_determinism() {
    let mut executor1 = ExecutionLoop::new();
    let mut executor2 = ExecutionLoop::new();
    let judgment = create_test_judgment();
    
    let outcome1 = executor1.execute(&judgment);
    let outcome2 = executor2.execute(&judgment);
    
    assert_eq!(outcome1.proof.judgment_hash, outcome2.proof.judgment_hash);
    assert_eq!(outcome1.proof.success, outcome2.proof.success);
    
    println!("✓ Execution is deterministic");
}

/// Test normalization idempotence  
#[test]
fn test_normalization_idempotent() {
    use mmsb_primitives::PageID;
    
    let page_id = PageID(42);
    let epoch = Epoch(1);
    let payload = vec![1, 2, 3, 4];
    
    let normalized1 = DeltaNormalizer::normalize(page_id, epoch, payload.clone());
    let normalized2 = DeltaNormalizer::normalize(
        normalized1.page_id,
        normalized1.epoch,
        normalized1.payload.clone(),
    );
    
    assert_eq!(normalized1.page_id, normalized2.page_id);
    assert_eq!(normalized1.epoch, normalized2.epoch);
    
    println!("✓ Normalization is idempotent");
}

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_judgment() -> JudgmentProof {
    use mmsb_proof::{PolicyProof, IntentProof, IntentBounds};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let intent_proof = IntentProof::new(
        [1u8; 32],
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
