//! Memory Engine
//!
//! Canonical truth owner responsible for producing proofs D, E, F.
//!
//! Authority:
//! - Owns canonical state and truth
//!
//! Produces:
//! - AdmissionProof (D)
//! - CommitProof (E)
//! - OutcomeProof (F)
//!
//! Consumes:
//! - JudgmentProof (C) via event or minimal witness

use std::{
    collections::HashSet,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use sha2::{Digest, Sha256};

use crate::{
    dag::DependencyGraph,
    delta::Delta,
    epoch::EpochCell,
    outcome::DagValidator,
    page::{PageAllocator, PageAllocatorConfig, PageLocation},
    tlog::TransactionLog,
};

use mmsb_events::{ExecutionRequested, MemoryCommitted};
use mmsb_proof::{
    AdmissionProof, CommitProof, Hash, JudgmentProof, OutcomeProof, Proof,
};

/// Configuration for initializing the MemoryEngine
pub struct MemoryEngineConfig {
    pub tlog_path: PathBuf,
    pub default_location: PageLocation,
}

/// Errors produced by the MemoryEngine
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

    #[error("Delta not found for hash")]
    DeltaNotFound,
}

/// Admission-stage errors (Proof D)
#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("Invalid JudgmentProof")]
    InvalidJudgmentProof,

    #[error("Stale epoch")]
    StaleEpoch,

    #[error("Execution already admitted (replay protection)")]
    AlreadyAdmitted,
}

/// Commit-stage errors (Proof E)
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("Failed to write to transaction log: {0}")]
    TlogWrite(#[source] std::io::Error),
}

/// Outcome-stage errors (Proof F)
#[derive(Debug, thiserror::Error)]
pub enum OutcomeError {
    #[error("Cycle detected in dependency graph")]
    CycleDetected,
}

/// Canonical Memory Engine
///
/// Owns truth and produces proofs D, E, F.
pub struct MemoryEngine {
    allocator: Arc<PageAllocator>,
    dag: Arc<RwLock<DependencyGraph>>,
    tlog: Arc<RwLock<TransactionLog>>,
    epoch: Arc<EpochCell>,
    admitted: Arc<RwLock<HashSet<Hash>>>,
    nonce_counter: AtomicU64,
}

impl MemoryEngine {
    /// Initialize a new MemoryEngine
    pub fn new(config: MemoryEngineConfig) -> Result<Self, MemoryEngineError> {
        let allocator = Arc::new(PageAllocator::from_config(PageAllocatorConfig {
            default_location: config.default_location,
            initial_capacity: 1024,
        }));

        let dag = Arc::new(RwLock::new(DependencyGraph::new()));

        let tlog = Arc::new(RwLock::new(
            TransactionLog::new(&config.tlog_path)
                .map_err(MemoryEngineError::TlogOpen)?,
        ));

        Ok(Self {
            allocator,
            dag,
            tlog,
            epoch: Arc::new(EpochCell::new(0)),
            admitted: Arc::new(RwLock::new(HashSet::new())),
            nonce_counter: AtomicU64::new(0),
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Admission Stage (D)
    // ─────────────────────────────────────────────────────────────────────

    fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, AdmissionError> {
        if !self.verify_judgment_proof(judgment_proof) {
            return Err(AdmissionError::InvalidJudgmentProof);
        }

        let current_epoch = self.epoch.load();
        if judgment_proof.timestamp < current_epoch.0 as u64 {
            return Err(AdmissionError::StaleEpoch);
        }

        let judgment_hash = judgment_proof.hash();
        let mut admitted = self.admitted.write();

        if admitted.contains(&judgment_hash) {
            return Err(AdmissionError::AlreadyAdmitted);
        }

        admitted.insert(judgment_hash);

        let nonce = self.nonce_counter.fetch_add(1, Ordering::Relaxed);

        Ok(AdmissionProof {
            judgment_proof_hash: judgment_hash,
            epoch: current_epoch.0 as u64,
            nonce,
        })
    }

    fn verify_judgment_proof(&self, proof: &JudgmentProof) -> bool {
        // Placeholder — delegate to mmsb-authenticate in production
        proof.approved && proof.hash() != Hash::default()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Commit Stage (E)
    // ─────────────────────────────────────────────────────────────────────

    fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta: &Delta,
    ) -> Result<CommitProof, CommitError> {
        self.epoch.increment();

        self.tlog
            .write()
            .append(admission, delta.clone())
            .map_err(CommitError::TlogWrite)?;

        Ok(CommitProof {
            admission_proof_hash: admission.hash(),
            delta_hash: delta.hash(),
            state_hash: self.compute_state_hash(),
            invariants_held: true,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Outcome Stage (F)
    // ─────────────────────────────────────────────────────────────────────

    fn record_outcome(
        &self,
        commit: &CommitProof,
    ) -> Result<OutcomeProof, OutcomeError> {
        let dag = self.dag.read();
        let validator = DagValidator::new(&dag);

        if validator.detect_cycles().has_cycle {
            return Err(OutcomeError::CycleDetected);
        }

        Ok(OutcomeProof {
            commit_proof_hash: commit.hash(),
            success: true,
            error_class: None,
            rollback_hash: None,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Event Entry Point
    // ─────────────────────────────────────────────────────────────────────

    pub fn handle_execution_requested(
        &mut self,
        event: ExecutionRequested,
    ) -> Result<MemoryCommitted, MemoryEngineError> {
        let admission = self.admit_execution(&event.judgment_proof)?;

        let delta = self
            .fetch_delta_by_hash(&event.delta_hash)
            .ok_or(MemoryEngineError::DeltaNotFound)?;

        let commit = self.commit_delta(&admission, &delta)?;
        let outcome = self.record_outcome(&commit)?;

        let epoch = self.epoch.load();
        let event_id = self.compute_event_hash(&admission, &commit, &outcome);

        let affected_page_ids = if event.affected_page_ids.is_empty() {
            vec![delta.page_id]
        } else {
            event.affected_page_ids
        };

        Ok(MemoryCommitted {
            event_id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            delta_hash: delta.hash(),
            epoch: epoch.0 as u64,
            snapshot_ref: None,
            admission_proof: admission,
            commit_proof: commit,
            outcome_proof: outcome,
            affected_page_ids,
        })
    }

    // ─────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────

    fn fetch_delta_by_hash(&self, _hash: &Hash) -> Option<Delta> {
        // TODO: Implement real lookup (storage / tlog)
        None
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
        // Placeholder — hash canonical state in production
        [0u8; 32]
    }
}

/// Temporary extension — move into `delta.rs`
impl Delta {
    pub fn hash(&self) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(&self.page_id.0.to_be_bytes());
        hasher.update(&self.epoch.0.to_be_bytes());
        hasher.update(&self.payload);
        hasher.finalize().into()
    }
}
