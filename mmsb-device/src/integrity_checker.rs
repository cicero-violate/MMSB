//! Delta integrity checks covering schema, orphan detection, and epoch ordering.

use super::device_registry::DeviceBufferRegistry;
use mmsb_primitives::{DeltaID, Epoch, PageID};
use std::collections::HashMap;
use std::sync::Arc;

// Note: Delta will be properly imported from mmsb-memory when full integration is ready
// For now, this module defines the interface mmsb-device needs from Delta

#[derive(Debug, Clone)]
pub enum IntegrityViolationKind {
    SchemaMismatch { mask_len: usize, payload_len: usize },
    OrphanDelta,
    EpochRegression { previous: Epoch, current: Epoch },
}

#[derive(Debug, Clone)]
pub struct IntegrityViolation {
    pub delta_id: DeltaID,
    pub page_id: PageID,
    pub kind: IntegrityViolationKind,
}

#[derive(Debug, Default)]
pub struct IntegrityReport {
    pub total: usize,
    pub violations: Vec<IntegrityViolation>,
}

impl IntegrityReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

pub struct DeltaIntegrityChecker {
    registry: Arc<DeviceBufferRegistry>,
    last_epoch: HashMap<PageID, Epoch>,
}

impl DeltaIntegrityChecker {
    pub fn new(registry: Arc<DeviceBufferRegistry>) -> Self {
        Self {
            registry,
            last_epoch: HashMap::new(),
        }
    }

    // Note: validate() method commented out until full Delta integration
    // pub fn validate(&mut self, deltas: &[Delta]) -> IntegrityReport { ... }
}
