//! Memory Engine - Canonical truth owner implementing D, E, F proof production
//!
//! Authority: Truth ownership only
//! Produces: AdmissionProof (D), CommitProof (E), OutcomeProof (F)
//! Consumes: JudgmentProof (C) — via event or minimal witness
//!
//! The MemoryEngine is the SOLE TRUTH AUTHORITY in MMSB.
//! It owns canonical time (epochs), structural invariants, mutation semantics,
//! replay protection, and the dependency graph (DAG).
//!
//! It does NOT know about:
//! - Execution runtime details
//! - Hardware allocation details (only logical page management)
//! - Scheduling / propagation
//!
//! Proof Chain:
//! C (JudgmentProof) → D (AdmissionProof) → E (CommitProof) → F (OutcomeProof)

use crate::dag::DependencyGraph;
use crate::delta::Delta;
use crate::epoch::EpochCell;
use crate::outcome::DagValidator;
use crate::page::{PageAllocator, PageAllocatorConfig};
use crate::tlog::TransactionLog;
use mmsb_events::{ExecutionRequested, MemoryCommitted};
use mmsb_proof::{
    AdmissionProof, CommitProof, Hash, JudgmentProof, OutcomeProof, Proof,
};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for initializing MemoryEngine
pub struct MemoryEngineConfig {
    pub tlog_path: PathBuf,
    pub default_location: crate::page::PageLocation,
}

/// Main errors that can occur during memory operations
#[derive(Debug, thiserror::Error)]
pub enum MemoryEngineError {
    #[error("Failed to open transaction log: {0}")]
    TlogOpen(#[source] std::io::Error),

    #[error("Admission failed: {0}")]
    Admission(#[from] AdmissionError),

    #[error("Commit failed: {0}")]
    Commit(#[from] CommitError),

    #[error("Outcome failed: {0}")]
    Outcome(#[from] OutcomeError),
}

#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("Invalid JudgmentProof")]
    InvalidJudgmentProof,
    #[error("Stale epoch")]
    StaleEpoch,
    #[error("Execution already admitted (replay protection)")]
    AlreadyAdmitted,
}

#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("Failed to write to transaction log: {0}")]
    TlogWrite(#[source] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum OutcomeError {
    #[error("Cycle detected in dependency graph")]
    CycleDetected,
}

/// Canonical Memory Engine — owns truth and produces proofs D, E, F
pub struct MemoryEngine {
    /// Logical page allocator (interface level, not physical)
    allocator: Arc<PageAllocator>,

    /// Canonical dependency graph (structural truth)
    dag: Arc<RwLock<DependencyGraph>>,

    /// Append-only transaction log
    tlog: Arc<RwLock<TransactionLog>>,

    /// Current canonical epoch
    epoch: Arc<EpochCell>,

    /// Replay protection — hashes of already admitted judgments
    admitted: Arc<RwLock<HashSet<Hash>>>,
}

impl MemoryEngine {
    /// Creates a new MemoryEngine instance
    pub fn new(config: MemoryEngineConfig) -> Result<Self, MemoryEngineError> {
        let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig {
            default_location: config.default_location,
        }));

        let dag = Arc::new(RwLock::new(DependencyGraph::new()));

        let tlog = Arc::new(RwLock::new(
            TransactionLog::new(&config.tlog_path).map_err(MemoryEngineError::TlogOpen)?,
        ));

        let epoch = Arc::new(EpochCell::new(0));
        let admitted = Arc::new(RwLock::new(HashSet::new()));

        Ok(Self {
            allocator,
            dag,
            tlog,
            epoch,
            admitted,
        })
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Admission Stage — produces AdmissionProof (D)
    // ────────────────────────────────────────────────────────────────────────────

    fn admit_execution(&self, judgment_proof: &JudgmentProof) -> Result<AdmissionProof, AdmissionError> {
        // Verify JudgmentProof (should delegate to mmsb-authenticate in full impl)
        if !self.verify_judgment_proof(judgment_proof) {
            return Err(AdmissionError::InvalidJudgmentProof);
        }

        let current_epoch = self.epoch.load();
        if judgment_proof.timestamp < current_epoch.0 as u64 {
            return Err(AdmissionError::StaleEpoch);
        }

        // Replay protection
        let judgment_hash = judgment_proof.hash();
        let mut admitted = self.admitted.write();
        if admitted.contains(&judgment_hash) {
            return Err(AdmissionError::AlreadyAdmitted);
        }
        admitted.insert(judgment_hash);

        Ok(AdmissionProof {
            judgment_proof_hash: judgment_hash,
            epoch: current_epoch.0 as u64,
            nonce: rand::random::<u64>(), // better nonce generation in production
        })
    }

    fn verify_judgment_proof(&self, judgment_proof: &JudgmentProof) -> bool {
        // Placeholder — in real implementation call mmsb-authenticate::VerifyProof
        judgment_proof.approved && judgment_proof.hash() != Hash::default()
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Commit Stage — produces CommitProof (E)
    // ────────────────────────────────────────────────────────────────────────────

    fn commit_delta(
        &self,
        admission_proof: &AdmissionProof,
        delta: &Delta,
    ) -> Result<CommitProof, CommitError> {
        // Apply delta logically (real impl would coordinate with allocator)
        // For now: simulate successful application

        let new_epoch = self.epoch.increment();

        // Record in transaction log (no JudgmentToken needed anymore)
        self.tlog
            .write()
            .append(admission_proof, delta.clone())
            .map_err(CommitError::TlogWrite)?;

        let state_hash = self.compute_state_hash();

        Ok(CommitProof {
            admission_proof_hash: admission_proof.hash(),
            delta_hash: delta.hash(),
            state_hash,
            invariants_held: true, // real validation would happen here
        })
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Outcome Stage — produces OutcomeProof (F)
    // ────────────────────────────────────────────────────────────────────────────

    fn record_outcome(&self, commit_proof: &CommitProof) -> Result<OutcomeProof, OutcomeError> {
        let dag = self.dag.read();
        let validator = DagValidator::new(&*dag);
        let report = validator.detect_cycles();

        if report.has_cycle {
            return Err(OutcomeError::CycleDetected);
        }

        Ok(OutcomeProof {
            commit_proof_hash: commit_proof.hash(),
            success: true,
            error_class: None,
            rollback_hash: None,
        })
    }

    // ────────────────────────────────────────────────────────────────────────────
    // Main public interface — event driven entry point
    // ────────────────────────────────────────────────────────────────────────────

    /// Primary entry point: handle ExecutionRequested event
    /// In full event-bus system this would be called by the event handler
    pub fn handle_execution_requested(
        &mut self,
        event: ExecutionRequested,
    ) -> Result<MemoryCommitted, MemoryEngineError> {
        // Extract needed information from event
        let judgment_proof = event.judgment_proof;

        // Stage D: Admission
        let admission_proof = self.admit_execution(&judgment_proof)?;

        // TODO: In real system → fetch delta from storage using event.delta_hash
        // For now: this is a placeholder — full version needs delta retrieval
        let delta = Delta::placeholder_from_hash(event.delta_hash); // ← MUST BE REPLACED

        // Stage E: Commit
        let commit_proof = self.commit_delta(&admission_proof, &delta)?;

        // Stage F: Outcome
        let outcome_proof = self.record_outcome(&commit_proof)?;

        // Create MemoryCommitted event
        let current_epoch = self.epoch.load();
        let event_id = self.compute_event_hash(&admission_proof, &commit_proof, &outcome_proof);

        Ok(MemoryCommitted {
            event_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            delta_hash: event.delta_hash,
            epoch: current_epoch.0 as u64,
            snapshot_ref: None,
            admission_proof,
            commit_proof,
            outcome_proof,
            // Crucial for executor propagation:
            affected_page_ids: event.affected_page_ids.unwrap_or_default(),
        })
    }

    fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(admission.hash());
        hasher.update(commit.hash());
        hasher.update(outcome.hash());
        hasher.finalize().into()
    }

    fn compute_state_hash(&self) -> Hash {
        // Placeholder — real impl would hash relevant state (DAG + pages)
        [0u8; 32]
    }
}

// Placeholder until real delta retrieval is implemented
impl Delta {
    pub fn placeholder_from_hash(_hash: Hash) -> Self {
        Delta::default() // ← REPLACE WITH ACTUAL STORAGE LOOKUP
    }
}
