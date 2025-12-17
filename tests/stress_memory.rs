use mmsb_core::page::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
use mmsb_core::utility::{MemoryMonitor, MemoryMonitorConfig};
use std::sync::Arc;

const TARGET_PAGES: usize = 1_000_000;
const SAMPLE_PAGES: usize = 2_048;
const PAGE_BYTES: usize = 1_024;
const GC_TARGET_MS: f64 = 3.0;
const GC_SAFETY_MS: f64 = 10.0;

#[test]
fn sampled_allocation_projects_under_budget() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    let monitor = MemoryMonitor::new(Arc::clone(&allocator));
    for idx in 0..SAMPLE_PAGES {
        allocator
            .allocate_raw(PageID(idx as u64 + 1), PAGE_BYTES, Some(PageLocation::Cpu))
            .expect("failed to allocate stress page");
    }
    let snapshot = monitor.snapshot();
    assert!(
        snapshot.avg_bytes_per_page <= PAGE_BYTES,
        "avg page size {} exceeded {} bytes target",
        snapshot.avg_bytes_per_page,
        PAGE_BYTES
    );
    let projected_total = snapshot
        .avg_bytes_per_page
        .saturating_mul(TARGET_PAGES);
    let one_gib = 1usize << 30;
    assert!(
        projected_total <= one_gib,
        "projected total {} bytes exceeded 1 GiB target",
        projected_total
    );
    println!(
        "METRIC:memory_snapshot avg={} total={} projected={}",
        snapshot.avg_bytes_per_page, snapshot.total_bytes, projected_total
    );
}

#[test]
fn incremental_gc_latency_stays_within_budget() {
    if cfg!(debug_assertions) {
        eprintln!("memory stress GC benchmark requires --release to assert timing");
        return;
    }
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    let monitor = MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 8 * 1024 * 1024,
            cold_page_age_limit: 0,
            incremental_batch_pages: 64,
        },
    );
    for idx in 0..(monitor.config().incremental_batch_pages * 4) {
        allocator
            .allocate_raw(PageID(idx as u64 + 1), PAGE_BYTES, Some(PageLocation::Cpu))
            .expect("failed to allocate GC page");
    }
    // Age pages so monitor considers them cold.
    for _ in 0..3 {
        monitor.snapshot();
    }
    let metrics = monitor
        .trigger_incremental_gc(monitor.config().incremental_batch_pages)
        .expect("GC should run under pressure");
    let gc_ms = metrics.duration.as_secs_f64() * 1_000.0;
    assert!(
        gc_ms <= GC_SAFETY_MS,
        "GC exceeded {} ms (target {:.1} ms)",
        gc_ms,
        GC_TARGET_MS
    );
    println!(
        "METRIC:memory_gc duration_ms={:.3} reclaimed={} pages={}",
        gc_ms,
        metrics.reclaimed_bytes,
        metrics.reclaimed_pages
    );
}

#[test]
fn fragmentation_probe_remains_stable() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    let monitor = MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 64 * 1024 * 1024,
            cold_page_age_limit: 2,
            incremental_batch_pages: 512,
        },
    );
    let mut live_pages = Vec::new();
    let mut next_page = 1u64;

    for iteration in 0..512 {
        // burst allocate varying chunk sizes to introduce fragmentation
        let burst = 32 + (iteration % 32) as usize;
        for _ in 0..burst {
            allocator
                .allocate_raw(PageID(next_page), PAGE_BYTES, Some(PageLocation::Cpu))
                .expect("allocate burst page");
            live_pages.push(PageID(next_page));
            next_page += 1;
        }
        // release older pages to keep working set bounded
        if live_pages.len() > 1024 {
            for _ in 0..(burst / 2) {
                if let Some(page_id) = live_pages.pop() {
                    allocator.free(page_id);
                }
            }
        }
        if iteration % 128 == 0 {
            monitor.snapshot();
        }
    }

    let snapshot = monitor.snapshot();
    assert!(
        snapshot.avg_bytes_per_page <= PAGE_BYTES,
        "fragmentation pushed avg page size {} bytes > {}",
        snapshot.avg_bytes_per_page,
        PAGE_BYTES
    );
    let estimate = snapshot.avg_bytes_per_page * snapshot.total_pages;
    assert!(
        estimate <= 1usize << 30,
        "estimated resident set {} bytes > 1 GiB",
        estimate
    );
    println!(
        "METRIC:memory_fragmentation total_pages={} estimate={}",
        snapshot.total_pages, estimate
    );
}

#[test]
#[ignore]
fn one_million_pages_under_1gb_full_run() {
    if cfg!(debug_assertions) {
        eprintln!("run `cargo test --release -- --ignored stress_memory` for the full 1M-page sweep");
        return;
    }
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for idx in 0..TARGET_PAGES {
        allocator
            .allocate_raw(PageID(idx as u64 + 1), PAGE_BYTES, Some(PageLocation::Cpu))
            .expect("failed to allocate large stress page");
    }
    let monitor = MemoryMonitor::new(Arc::clone(&allocator));
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.total_pages, TARGET_PAGES);
    assert!(
        snapshot.total_bytes <= 1usize << 30,
        "total bytes {} exceeded 1 GiB budget",
        snapshot.total_bytes
    );
    assert!(
        snapshot.avg_bytes_per_page <= PAGE_BYTES,
        "avg page size {} exceeded {} bytes",
        snapshot.avg_bytes_per_page,
        PAGE_BYTES
    );
    println!(
        "METRIC:memory_full_run total={} avg={}",
        snapshot.total_bytes, snapshot.avg_bytes_per_page
    );
}
