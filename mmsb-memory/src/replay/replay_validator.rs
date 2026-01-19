//! Deterministic replay validator built on top of checkpoint snapshots.

use crate::page::{PageAllocator, PageSnapshotData, PageError};
use mmsb_primitives::PageID;
use crate::tlog::TransactionLog;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ReplayCheckpoint {
    pub id: usize,
    pub log_offset: u64,
    pub snapshot: Vec<PageSnapshotData>,
}

#[derive(Debug)]
pub struct ReplayReport {
    pub checkpoint_id: usize,
    pub divergence: f64,
    pub violations: Vec<PageID>,
    pub max_delta: u8,
    pub log_offset: u64,
}

impl ReplayReport {
    pub fn passed(&self, threshold: f64) -> bool {
        self.divergence <= threshold
    }
}

#[derive(Debug)]
pub struct ReplayValidator {
    checkpoints: Vec<ReplayCheckpoint>,
    threshold: f64,
}

impl ReplayValidator {
    pub fn new(threshold: f64) -> Self {
        Self {
            checkpoints: Vec::new(),
            threshold,
        }
    }

    pub fn record_checkpoint(
        &mut self,
        allocator: &PageAllocator,
        tlog: &TransactionLog,
    ) -> std::io::Result<usize> {
        let snapshot = allocator.snapshot_pages();
        let offset = tlog.current_offset()?;
        let id = self.checkpoints.len();
        self.checkpoints.push(ReplayCheckpoint {
            id,
            log_offset: offset,
            snapshot,
        });
        Ok(id)
    }

    pub fn checkpoint(&self, idx: usize) -> Option<&ReplayCheckpoint> {
        self.checkpoints.get(idx)
    }

    pub fn validate_allocator(
        &self,
        checkpoint_id: usize,
        allocator: &PageAllocator,
    ) -> Result<ReplayReport, PageError> {
        let snapshot = allocator.snapshot_pages();
        Ok(self.compare_with_snapshot(checkpoint_id, &snapshot)?)
    }

    pub fn compare_with_snapshot(
        &self,
        checkpoint_id: usize,
        snapshot: &[PageSnapshotData],
    ) -> Result<ReplayReport, PageError> {
        let checkpoint = self
            .checkpoint(checkpoint_id)
            .ok_or(PageError::MetadataDecode("invalid checkpoint id".to_string()))?;
        Ok(compare_snapshots(checkpoint, snapshot))
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

fn compare_snapshots(
    checkpoint: &ReplayCheckpoint,
    current: &[PageSnapshotData],
) -> ReplayReport {
    let mut baseline = HashMap::new();
    for page in &checkpoint.snapshot {
        baseline.insert(page.page_id, page);
    }

    let mut divergence = 0.0f64;
    let mut max_delta = 0u8;
    let mut violations = Vec::new();

    for page in current {
        if let Some(reference) = baseline.remove(&page.page_id) {
            let (delta, local_max) = l2_distance(&reference.data, &page.data);
            divergence += delta;
            max_delta = max_delta.max(local_max);
        } else {
            violations.push(page.page_id);
        }
    }

    for missing in baseline.keys() {
        violations.push(*missing);
    }

    ReplayReport {
        checkpoint_id: checkpoint.id,
        divergence: divergence.sqrt(),
        violations,
        max_delta,
        log_offset: checkpoint.log_offset,
    }
}

fn l2_distance(reference: &[u8], candidate: &[u8]) -> (f64, u8) {
    let len = reference.len().min(candidate.len());
    let mut acc = 0.0f64;
    let mut max_delta = 0u8;
    for idx in 0..len {
        let delta = reference[idx] as i32 - candidate[idx] as i32;
        max_delta = max_delta.max(delta.unsigned_abs() as u8);
        acc += (delta * delta) as f64;
    }
    if reference.len() != candidate.len() {
        max_delta = u8::MAX;
        acc += ((reference.len() as i64 - candidate.len() as i64).abs() as f64).powi(2);
    }
    (acc, max_delta)
}

#[cfg(test)]
mod tests {
    use super::ReplayValidator;
    use crate::page::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
    use crate::page::tlog::TransactionLog;
    use crate::types::Epoch;
    use std::fs;
    use std::path::PathBuf;

    fn temp_log_path() -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("mmsb_replay_{}.log", rand_suffix()));
        path
    }

    fn rand_suffix() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    #[test]
    fn checkpoint_validation_detects_divergence() {
        let path = temp_log_path();
        let log = TransactionLog::new(&path).unwrap();
        let allocator = PageAllocator::new(PageAllocatorConfig::default());
        allocator
            .allocate_raw(PageID(1), 4, Some(PageLocation::Cpu))
            .unwrap();
        allocator
            .allocate_raw(PageID(2), 4, Some(PageLocation::Cpu))
            .unwrap();

        {
            let page1 = allocator.acquire_page(PageID(1)).unwrap();
            let page2 = allocator.acquire_page(PageID(2)).unwrap();
            unsafe {
                (*page1).data_mut_slice().copy_from_slice(b"\x01\x02\x03\x04");
                (*page1).set_epoch(Epoch(5));
                (*page2).data_mut_slice().copy_from_slice(b"\x05\x06\x07\x08");
            }
        }

        let mut validator = ReplayValidator::new(1e-9);
        let checkpoint_id = validator.record_checkpoint(&allocator, &log).unwrap();

        {
            let page1 = allocator.acquire_page(PageID(1)).unwrap();
            unsafe {
                (*page1).data_mut_slice()[0] = 0xAA;
            }
        }

        let report = validator.validate_allocator(checkpoint_id, &allocator).unwrap();
        assert!(!report.passed(validator.threshold()));
        assert_eq!(report.violations.len(), 0);

        fs::remove_file(path).ok();
    }
}
