//! Admission Proof Production - Canonical D proof in mmsb-memory

use mmsb_proof::{AdmissionProof, JudgmentProof, Hash, Proof};
use mmsb_events::ExecutionRequested;
use sha2::{Digest, Sha256};

use crate::epoch::EpochCell;
use crate::tlog::TransactionLog;
use crate::dag::DependencyGraph;

use std::sync::Arc;
use parking_lot::RwLock;

/// Admission Gate - Produces AdmissionProof (D) from JudgmentProof (C)
pub struct AdmissionGate {
    epoch: Arc<EpochCell>,
    admitted: Arc<RwLock<std::collections::HashSet<Hash>>>,  // Replay protection
    dag: Arc<RwLock<DependencyGraph>>,  // For DAG snapshot if needed
}

impl AdmissionGate {
    pub fn new(epoch: Arc<EpochCell>, dag: Arc<RwLock<DependencyGraph>>) -> Self {
        Self {
            epoch,
            admitted: Arc::new(RwLock::new(std::collections::HashSet::new())),
            dag,
        }
    }

    /// Produce AdmissionProof (D) from ExecutionRequested event
    pub fn produce_admission_proof(
        &self,
        event: &ExecutionRequested,
    ) -> Result<AdmissionProof, String> {
        let judgment_proof = &event.judgment_proof;

        // Verify JudgmentProof (placeholder - delegate to mmsb-authenticate)
        if !self.verify_judgment_proof(judgment_proof) {
            return Err("Invalid JudgmentProof".to_string());
        }

        let current_epoch = self.epoch.load().0 as u64;
        if judgment_proof.timestamp < current_epoch {
            return Err("Stale epoch".to_string());
        }

        let judgment_hash = judgment_proof.hash();
        let mut admitted = self.admitted.write();
        if admitted.contains(&judgment_hash) {
            return Err("Already admitted (replay protection)".to_string());
        }
        admitted.insert(judgment_hash);

        // Optional: Compute DAG snapshot hash if required
        let dag_snapshot_hash = Some(self.dag.read().compute_snapshot_hash());

        Ok(AdmissionProof {
            judgment_proof_hash: judgment_hash,
            epoch: current_epoch,
            nonce: rand::random::<u64>(),  // Better nonce in production
        })
    }

    fn verify_judgment_proof(&self, judgment_proof: &JudgmentProof) -> bool {
        // Placeholder - real verification in mmsb-authenticate
        judgment_proof.approved
    }
}

// Remove all file I/O, JSON, policy loading, streams, MmsbAdmissionProof, etc.
// Those belong in mmsb-policy or mmsb-storage, not memory
