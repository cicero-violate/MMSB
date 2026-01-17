//! Memory Engine - Canonical truth owner implementing D, E, F proof production
//!
//! Authority: Truth ownership only
//! Produces: AdmissionProof (D), CommitProof (E), OutcomeProof (F)
//! Consumes: JudgmentProof (C) — via event or minimal witness
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
use mmsb_primitives::PageID;  // Add if needed for affected_page_ids
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
#[error("Delta not found for hash")]
DeltaNotFound,
}
/// Admission-specific errors
#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
#[error("Invalid JudgmentProof")]
InvalidJudgmentProof,
#[error("Stale epoch")]
StaleEpoch,
#[error("Execution already admitted (replay protection)")]
AlreadyAdmitted,
}
/// Commit-specific errors
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
#[error("Failed to write to transaction log: {0}")]
TlogWrite(#[source] std::io::Error),
}
/// Outcome-specific errors
#[derive(Debug, thiserror::Error)]
pub enum OutcomeError {
#[error("Cycle detected in dependency graph")]
CycleDetected,
}
/// Canonical Memory Engine — owns truth and produces proofs D, E, F
pub struct MemoryEngine {
allocator: Arc<PageAllocator>,
dag: Arc<RwLock<DependencyGraph>>,
tlog: Arc<RwLock<TransactionLog>>,
epoch: Arc<EpochCell>,
admitted: Arc<RwLock<HashSet<Hash>>>,
// Simple counter for deterministic nonce (replace with rand later if needed)
nonce_counter: std::sync::atomic::AtomicU64,
}
impl MemoryEngine {
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
nonce_counter: std::sync::atomic::AtomicU64::new(0),
})
}
// ────────────────────────────────────────────────────────────────────────────
// Admission Stage — produces AdmissionProof (D)
// ────────────────────────────────────────────────────────────────────────────
fn admit_execution(&self, judgment_proof: &JudgmentProof) -> Result<AdmissionProof, AdmissionError> {
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
let nonce = self.nonce_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
Ok(AdmissionProof {
judgment_proof_hash: judgment_hash,
epoch: current_epoch.0 as u64,
nonce,
})
}
fn verify_judgment_proof(&self, judgment_proof: &JudgmentProof) -> bool {
// Placeholder — delegate to mmsb-authenticate in production
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
let new_epoch = self.epoch.increment();
// Append using AdmissionProof
self.tlog
.write()
.append(admission_proof, delta.clone())
.map_err(CommitError::TlogWrite)?;
let state_hash = self.compute_state_hash();
Ok(CommitProof {
admission_proof_hash: admission_proof.hash(),
delta_hash: delta.hash(),
state_hash,
invariants_held: true,
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
// Main entry point — should be called by event handler
// ────────────────────────────────────────────────────────────────────────────
pub fn handle_execution_requested(
&mut self,
event: ExecutionRequested,
) -> Result<MemoryCommitted, MemoryEngineError> {
let judgment_proof = &event.judgment_proof;
// Stage D: Admission
let admission_proof = self.admit_execution(judgment_proof)?;
// Fetch delta by hash (placeholder — replace with real)
let delta = self.fetch_delta_by_hash(&event.delta_hash)
.ok_or(MemoryEngineError::DeltaNotFound)?;
// Stage E: Commit
let commit_proof = self.commit_delta(&admission_proof, &delta)?;
// Stage F: Outcome
let outcome_proof = self.record_outcome(&commit_proof)?;
let current_epoch = self.epoch.load();
let event_id = self.compute_event_hash(&admission_proof, &commit_proof, &outcome_proof);
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
epoch: current_epoch.0 as u64,
snapshot_ref: None,
admission_proof,
commit_proof,
outcome_proof,
affected_page_ids,
})
}
// Placeholder for delta fetch — implement real (storage/tlog lookup)
fn fetch_delta_by_hash(&self, _hash: &Hash) -> Option<Delta> {
// TODO: Real implementation
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
// Placeholder — hash real state in production
[0u8; 32]
}
}
// Temporary extension to Delta (move to delta.rs)
impl Delta {
pub fn hash(&self) -> Hash {
let mut hasher = Sha256::new();
hasher.update(&self.page_id.0.to_be_bytes());
hasher.update(&self.epoch.0.to_be_bytes());
hasher.update(&self.payload);
hasher.finalize().into()
}
}
