//! Delta integrity checks covering schema, orphan detection, and epoch ordering.

use crate::device::DeviceRegistry;
use crate::delta::{Delta, DeltaID};
use crate::epoch::Epoch;
use mmsb_primitives::PageID;
use std::collections::HashMap;
use std::sync::Arc;

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
    registry: Arc<DeviceRegistry>,
    last_epoch: HashMap<PageID, Epoch>,
}

impl DeltaIntegrityChecker {
    pub fn new(registry: Arc<DeviceRegistry>) -> Self {
        Self {
            registry,
            last_epoch: HashMap::new(),
        }
    }

    pub fn validate(&mut self, deltas: &[Delta]) -> IntegrityReport {
        let mut report = IntegrityReport {
            total: deltas.len(),
            violations: Vec::new(),
        };

        for delta in deltas {
            if !self.registry.contains(delta.page_id) {
                report.violations.push(IntegrityViolation {
                    delta_id: delta.delta_id,
                    page_id: delta.page_id,
                    kind: IntegrityViolationKind::OrphanDelta,
                });
                continue;
            }

            if !schema_valid(delta) {
                report.violations.push(IntegrityViolation {
                    delta_id: delta.delta_id,
                    page_id: delta.page_id,
                    kind: IntegrityViolationKind::SchemaMismatch {
                        mask_len: delta.mask.len(),
                        payload_len: delta.payload.len(),
                    },
                });
                continue;
            }

            let last_epoch = self
                .last_epoch
                .entry(delta.page_id)
                .or_insert(Epoch(0));
            if delta.epoch.0 < last_epoch.0 {
                report.violations.push(IntegrityViolation {
                    delta_id: delta.delta_id,
                    page_id: delta.page_id,
                    kind: IntegrityViolationKind::EpochRegression {
                        previous: *last_epoch,
                        current: delta.epoch,
                    },
                });
            } else {
                *last_epoch = delta.epoch;
            }
        }

        report
    }
}

fn schema_valid(delta: &Delta) -> bool {
    if delta.is_sparse {
        let changed = delta.mask.iter().filter(|flag| **flag).count();
        changed == delta.payload.len()
    } else {
        delta.mask.len() == delta.payload.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{DeltaIntegrityChecker, IntegrityViolationKind};
    use crate::page::{Delta, DeltaID, DeviceRegistry, Page, PageID, PageLocation, Source};
    use crate::types::Epoch;
    use std::sync::Arc;

    fn page(page_id: u64) -> Arc<Page> {
        Arc::new(Page::new(PageID(page_id), 4, PageLocation::Cpu).unwrap())
    }

    fn delta(delta_id: u64, page_id: u64, epoch: u32, payload: &[u8]) -> Delta {
        Delta {
            delta_id: DeltaID(delta_id),
            page_id: PageID(page_id),
            epoch: Epoch(epoch),
            mask: payload.iter().map(|_| true).collect(),
            payload: payload.to_vec(),
            is_sparse: false,
            timestamp: 0,
            source: Source("test".into()),
            intent_metadata: None,
        }
    }

    #[test]
    fn detects_orphan_and_epoch_errors() {
        let registry = Arc::new(DeviceRegistry::default());
        registry.insert(page(1));
        let mut checker = DeltaIntegrityChecker::new(Arc::clone(&registry));
        let deltas = vec![
            delta(1, 1, 1, b"abc"),
            delta(2, 1, 0, b"abc"),
            delta(3, 2, 1, b"abc"),
        ];
        let report = checker.validate(&deltas);
        assert_eq!(report.total, 3);
        assert_eq!(report.violations.len(), 2);
        matches!(report.violations[0].kind, IntegrityViolationKind::EpochRegression { .. });
        matches!(report.violations[1].kind, IntegrityViolationKind::OrphanDelta);
    }
}
