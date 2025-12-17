use mmsb_core::dag::{EdgeType, ShadowPageGraph};
use mmsb_core::page::{
    Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source,
};
use mmsb_core::propagation::{ThroughputEngine, TickOrchestrator};
use mmsb_core::types::Epoch;
use mmsb_core::utility::{MemoryMonitor, MemoryMonitorConfig};
use std::sync::Arc;

fn delta(id: u64, page: u64) -> Delta {
    Delta {
        delta_id: DeltaID(id),
        page_id: PageID(page),
        epoch: Epoch(id as u32),
        mask: vec![true; 8],
        payload: vec![id as u8; 8],
        is_sparse: false,
        timestamp: id,
        source: Source("tick".into()),
        intent_metadata: None,
    }
}

#[test]
fn tick_latency_stays_within_budget() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for id in 1..=4 {
        allocator
            .allocate_raw(PageID(id), 8, Some(PageLocation::Cpu))
            .unwrap();
    }
    let throughput = ThroughputEngine::new(Arc::clone(&allocator), 2, 16);
    let graph = Arc::new(ShadowPageGraph::default());
    graph.add_edge(PageID(1), PageID(2), EdgeType::Data);
    graph.add_edge(PageID(2), PageID(3), EdgeType::Data);
    let memory = Arc::new(MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 1024 * 8,
            ..MemoryMonitorConfig::default()
        },
    ));
    let orchestrator = TickOrchestrator::new(throughput, graph, memory);
    let deltas: Vec<_> = (0..64).map(|idx| delta(idx, (idx % 4) + 1)).collect();
    let metrics = orchestrator.execute_tick(deltas).unwrap();
    assert!(metrics.total.as_millis() < 16);
}
