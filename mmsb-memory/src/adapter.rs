//! Memory Bus Adapters
//!
//! Implements mmsb-events bus traits for MemoryEngine.
//! This is the presentation layer that exposes MemoryEngine via standard bus protocols.

use std::sync::Arc;
use parking_lot::Mutex;

use mmsb_events::{StateBus, MemoryReader, Fact, MemoryCommitted};
use mmsb_primitives::{Hash, PageID, DeltaID};
use mmsb_proof::{AdmissionProof, CommitProof, JudgmentProof};
use tokio::sync::broadcast;

use crate::memory_engine::MemoryEngine;
use crate::delta::{Delta, Source};
use crate::epoch::Epoch;
use crate::notifier::CommitNotifier;

/// Adapter wrapping MemoryEngine to implement bus traits
///
/// This provides both read (MemoryReader) and write (StateBus) access
/// to the canonical memory engine.
pub struct MemoryAdapter {
    engine: Arc<Mutex<MemoryEngine>>,
    notifier: Arc<CommitNotifier>,
}

impl MemoryAdapter {
    /// Create a new adapter wrapping a MemoryEngine
    pub fn new(engine: Arc<Mutex<MemoryEngine>>, notifier: Arc<CommitNotifier>) -> Self {
        Self { engine, notifier }
    }
    
    /// Get a reference to the underlying engine (for direct access if needed)
    pub fn engine(&self) -> &Arc<Mutex<MemoryEngine>> {
        &self.engine
    }
}

impl StateBus for MemoryAdapter {
    fn admit(&mut self, judgment_proof: JudgmentProof) -> Result<AdmissionProof, mmsb_events::state_bus::AdmissionError> {
        self.engine
            .lock()
            .admit_execution(&judgment_proof)
            .map_err(|e| mmsb_events::state_bus::AdmissionError(format!("{}", e)))
    }
    
    fn commit(&mut self, fact: Fact) -> Result<CommitProof, mmsb_events::state_bus::CommitError> {
        // Convert Fact to Delta
        let current_epoch = self.engine.lock().current_epoch();
        let payload_len = fact.content.len();
        
        let delta = Delta {
            delta_id: DeltaID(fact.timestamp), // Use timestamp as delta ID
            page_id: PageID(0), // TODO: Extract from fact metadata
            epoch: Epoch((current_epoch).try_into().unwrap()),
            mask: vec![true; payload_len], // Dense - all bytes changed
            payload: fact.content,
            is_sparse: false,
            timestamp: fact.timestamp,
            source: Source("fact_commit".to_string()),
            intent_metadata: None,
        };
        
        // Create a minimal admission proof for the commit
        // In production, this should come from the judgment flow
        let admission = AdmissionProof {
            judgment_proof_hash: fact.fact_hash,
            epoch: current_epoch,
            nonce: 0,
        };
        
        self.engine
            .lock()
            .commit_delta(&admission, &delta)
            .map_err(|e| mmsb_events::state_bus::CommitError(format!("{}", e)))
    }
    
    fn broadcast_delta(&self, _delta: mmsb_events::state_bus::Delta) {
        // Optional: Hook for observers/notifiers
        // For now, no-op. In production, could notify subscribers.
    }
}

impl MemoryReader for MemoryAdapter {
    fn check_admitted(&self, judgment_hash: Hash) -> bool {
        self.engine.lock().check_admitted(&judgment_hash)
    }
    
    fn current_epoch(&self) -> u64 {
        self.engine.lock().current_epoch()
    }
    
    fn get_delta(&self, delta_hash: Hash) -> Option<Vec<u8>> {
        self.engine.lock()
            .fetch_delta_by_hash(&delta_hash)
            .map(|delta| delta.payload)
    }
    
    fn query_page(&self, _page_id: PageID) -> Option<Vec<u8>> {
        // TODO: Implement page query via allocator
        // self.engine.lock().allocator.read_page(page_id)
        None
    }
    
    fn delta_exists(&self, delta_hash: Hash) -> bool {
        self.engine.lock()
            .fetch_delta_by_hash(&delta_hash)
            .is_some()
    }
    
    fn subscribe_commits(&self) -> broadcast::Receiver<MemoryCommitted> {
        self.notifier.subscribe()
    }
}
