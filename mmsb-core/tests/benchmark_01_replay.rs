// Use the public prelude API
use mmsb_core::prelude::{
    PageAllocator, PageAllocatorConfig, PageID, PageLocation, ReplayValidator, TransactionLog,
};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_log_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "mmsb_replay_{}.log",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path
}

#[test]
fn replay_validator_divergence_under_threshold() {
    let path = temp_log_path();
    let log = TransactionLog::new(&path).unwrap();
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    allocator
        .allocate_raw(PageID(1), 1024, Some(PageLocation::Cpu))
        .unwrap();
    let mut validator = ReplayValidator::new(1e-9);
    let checkpoint = validator.record_checkpoint(&allocator, &log).unwrap();
    let report = validator.validate_allocator(checkpoint, &allocator).unwrap();
    assert!(report.passed(validator.threshold()));
    std::fs::remove_file(path).ok();
}
