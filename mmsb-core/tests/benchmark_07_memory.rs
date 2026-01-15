// Use the public prelude API
use mmsb_core::prelude::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
use mmsb_core::prelude::{MemoryMonitor, MemoryMonitorConfig};
use std::sync::Arc;

#[test]
fn memory_monitor_enforces_limits() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for id in 1..=128 {
        allocator
            .allocate_raw(PageID(id), 1024, Some(PageLocation::Cpu))
            .unwrap();
    }
    let monitor = MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 64 * 1024,
            cold_page_age_limit: 0,
            incremental_batch_pages: 16,
        },
    );
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.total_pages, 128);
    assert!(snapshot.total_bytes <= 128 * 1024);
    let metrics = monitor.trigger_incremental_gc(32).unwrap();
    assert!(metrics.reclaimed_pages <= 32);
}
