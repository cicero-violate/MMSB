// Use the public prelude API
use mmsb_core::prelude::{Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source};
use mmsb_core::prelude::ThroughputEngine;
use mmsb_core::prelude::Epoch;
use std::sync::Arc;

fn make_delta(id: u64, page: u64) -> Delta {
    Delta {
        delta_id: DeltaID(id),
        page_id: PageID(page),
        epoch: Epoch(id as u32),
        mask: vec![true; 16],
        payload: vec![id as u8; 16],
        is_sparse: false,
        timestamp: id,
        source: Source("bench".into()),
        intent_metadata: None,
    }
}

#[test]
fn throughput_engine_exceeds_minimum_rate() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for id in 1..=8 {
        allocator
            .allocate_raw(PageID(id), 16, Some(PageLocation::Cpu))
            .unwrap();
    }
    let engine = ThroughputEngine::new(Arc::clone(&allocator), 2, 32);
    let deltas: Vec<_> = (0..512).map(|idx| make_delta(idx, (idx % 8) + 1)).collect();
    let metrics = engine.process_parallel(deltas).unwrap();
    assert_eq!(metrics.processed, 512);
    assert!(metrics.throughput > 50_000.0);
}
