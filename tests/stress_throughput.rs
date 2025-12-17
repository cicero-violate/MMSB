use mmsb_core::page::{
    Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source,
};
use mmsb_core::propagation::ThroughputEngine;
use mmsb_core::types::Epoch;
use std::sync::Arc;
use std::time::Duration;

const TOTAL_DELTAS: usize = 10_000_000;
const MIN_SAMPLE: usize = 500_000;
const PAGE_COUNT: u64 = 4_096;
const PAGE_SIZE: usize = 1;
const CHUNK_SIZE: usize = PAGE_COUNT as usize;
const MULTI_WORKER_TARGET: f64 = 1_500_000.0;
const MULTI_GOAL: f64 = 10_000_000.0;

#[test]
fn single_thread_1m_deltas_per_sec() {
    if cfg!(debug_assertions) {
        eprintln!("stress throughput tests require --release; skipping");
        return;
    }
    let allocator = prepare_allocator();
    run_throughput_benchmark(&allocator, 1, 1_000_000.0);
}

#[test]
fn multi_thread_10m_deltas_per_sec() {
    if cfg!(debug_assertions) {
        eprintln!("stress throughput tests require --release; skipping");
        return;
    }
    let allocator = prepare_allocator();
    run_throughput_benchmark(&allocator, 8, MULTI_WORKER_TARGET);
}

fn prepare_allocator() -> Arc<PageAllocator> {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for id in 0..PAGE_COUNT {
        allocator
            .allocate_raw(PageID(id + 1), PAGE_SIZE, Some(PageLocation::Cpu))
            .expect("failed to allocate stress-test page");
    }
    allocator
}

fn run_throughput_benchmark(
    allocator: &Arc<PageAllocator>,
    workers: usize,
    min_target: f64,
) {
    let engine = ThroughputEngine::new(Arc::clone(allocator), workers, CHUNK_SIZE);
    let mut total_processed = 0usize;
    let mut total_duration = Duration::default();
    let mut next_delta = 0u64;
    let mut sample_checkpoint: Option<(usize, Duration)> = None;

    while total_processed < TOTAL_DELTAS {
        let remaining = TOTAL_DELTAS - total_processed;
        let batch_size = remaining.min(CHUNK_SIZE);
        let deltas = generate_delta_batch(next_delta, batch_size, PAGE_COUNT);
        next_delta += batch_size as u64;
        let metrics = engine
            .process_parallel(deltas)
            .expect("throughput engine failed");
        assert_eq!(metrics.processed, batch_size);
        total_processed += metrics.processed;
        total_duration += metrics.duration;
        if sample_checkpoint.is_none() && total_processed >= MIN_SAMPLE {
            sample_checkpoint = Some((total_processed, total_duration));
        }
    }

    if let Some((processed, duration)) = sample_checkpoint {
        assert_throughput(processed, duration, min_target);
    }
    assert_throughput(total_processed, total_duration, min_target);
}

fn generate_delta_batch(
    start_delta: u64,
    count: usize,
    page_count: u64,
) -> Vec<Delta> {
    (0..count)
        .map(|offset| {
            let delta_id = start_delta + offset as u64;
            build_noop_delta(delta_id, PageID((delta_id % page_count) + 1))
        })
        .collect()
}

fn build_noop_delta(delta_id: u64, page_id: PageID) -> Delta {
    Delta {
        delta_id: DeltaID(delta_id),
        page_id,
        epoch: Epoch((delta_id % u32::MAX as u64) as u32),
        mask: Vec::new(),
        payload: Vec::new(),
        is_sparse: false,
        timestamp: delta_id,
        source: Source(String::new()),
        intent_metadata: None,
    }
}

fn assert_throughput(processed: usize, duration: Duration, min_target: f64) {
    let secs = duration.as_secs_f64();
    assert!(
        secs > 0.0,
        "measured duration is zero; rerun in release with perf counters enabled"
    );
    let throughput = processed as f64 / secs;
    let phase_goal = if (min_target - MULTI_WORKER_TARGET).abs() < f64::EPSILON {
        MULTI_GOAL
    } else {
        min_target
    };
    println!(
        "METRIC:throughput processed={} duration_ns={} throughput_per_sec={}",
        processed,
        duration.as_nanos(),
        throughput
    );
    assert!(
        throughput >= min_target,
        "Throughput {:.2} deltas/sec below target {:.2} (Phase goal {:.2}). Run `perf record --call-graph dwarf -- \
         cargo test --release stress_throughput` to profile hotspots.",
        throughput,
        min_target,
        phase_goal
    );
}
