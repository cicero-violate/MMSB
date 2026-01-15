// Use the public prelude API
use mmsb_core::prelude::{EdgeType, DependencyGraph, StructuralOp};
use mmsb_core::prelude::{
    Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source,
};
use mmsb_core::prelude::{ThroughputEngine, TickOrchestrator};
use mmsb_core::prelude::Epoch;
use mmsb_core::prelude::{MemoryMonitor, MemoryMonitorConfig};
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
    let mut dag = DependencyGraph::new();
    let ops = vec![
        StructuralOp::AddEdge { from: PageID(1), to: PageID(2), edge_type: EdgeType::Data },
        StructuralOp::AddEdge { from: PageID(2), to: PageID(3), edge_type: EdgeType::Data },
    ];
    dag.apply_ops(&ops);
    let dag = Arc::new(dag);
    let memory = Arc::new(MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 1024 * 8,
            ..MemoryMonitorConfig::default()
        },
    ));
    let orchestrator = TickOrchestrator::new(throughput, dag, memory);
    let deltas: Vec<_> = (0..64).map(|idx| delta(idx, (idx % 4) + 1)).collect();
    let metrics = orchestrator.execute_tick(deltas).unwrap();
    assert!(metrics.total.as_millis() < 16);
}
