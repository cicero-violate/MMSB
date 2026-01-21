//! Delta Normalizer - converts execution results to normalized deltas
//!
//! Authority: NONE (pure derivation)

use mmsb_memory::delta::Delta;
use mmsb_memory::epoch::Epoch;
use mmsb_primitives::PageID;

/// Normalized delta ready for admission
#[derive(Debug, Clone)]
pub struct NormalizedDelta {
    pub page_id: PageID,
    pub epoch: Epoch,
    pub payload: Vec<u8>,
    pub dependencies: Vec<PageID>,
}

impl NormalizedDelta {
    pub fn to_delta(&self) -> Delta {
        use std::time::{SystemTime, UNIX_EPOCH};
        Delta {
            delta_id: mmsb_primitives::DeltaID(0), // TODO: generate proper ID
            page_id: self.page_id,
            epoch: self.epoch,
            mask: vec![],
            payload: self.payload.clone(),
            is_sparse: false,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            source: mmsb_memory::delta::Source("Executor".to_string()),
            intent_metadata: None,
        }
    }
}

/// DeltaNormalizer - pure transformation logic
pub struct DeltaNormalizer;

impl DeltaNormalizer {
    /// Normalize a proposed delta from executor
    /// CRITICAL: This function NEVER mutates state - pure derivation only
    pub fn normalize(
        page_id: PageID,
        epoch: Epoch,
        payload: Vec<u8>,
    ) -> NormalizedDelta {
        // Pure transformation logic
        // In production: type checking, schema validation, normalization
        
        NormalizedDelta {
            page_id,
            epoch,
            payload,
            dependencies: vec![], // TODO: derive from DAG
        }
    }

    /// Derive secondary deltas from dependency graph
    pub fn derive_secondary_deltas(
        _primary: &NormalizedDelta,
    ) -> Vec<NormalizedDelta> {
        // TODO: Walk dependency graph, compute dependent updates
        vec![]
    }
}
