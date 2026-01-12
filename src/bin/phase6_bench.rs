// Use the public prelude API
use mmsb_core::prelude::*;

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn Error>> {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    let page_count: u64 = 64;
    for id in 1..=page_count {
        allocator
            .allocate_raw(PageID(id), 4096, Some(PageLocation::Cpu))
            .expect("allocate page");
    }

    let deltas = build_deltas(20_000, page_count as u64);
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let throughput_engine = ThroughputEngine::new(Arc::clone(&allocator), worker_count, 1024);
    let throughput_metrics = throughput_engine.process_parallel(deltas.clone())?;

    let graph = Arc::new(build_graph(page_count));
    let memory_monitor: Arc<dyn MemoryPressureHandler> = Arc::new(MemoryMonitor::with_config(
        Arc::clone(&allocator),
        MemoryMonitorConfig {
            gc_threshold_bytes: 64 * 4096,
            ..MemoryMonitorConfig::default()
        },
    ));
    let tick_throughput = ThroughputEngine::new(Arc::clone(&allocator), worker_count, 1024);
    let orchestrator = TickOrchestrator::new(tick_throughput, Arc::clone(&graph), memory_monitor);
    let tick_metrics = orchestrator.execute_tick(deltas)?;

    write_report(&throughput_metrics, &tick_metrics, worker_count)?;
    println!(
        "Phase 6 benchmarks captured: throughput={:.2} deltas/sec total_tick_ms={:.3}",
        throughput_metrics.throughput,
        tick_metrics.total.as_secs_f64() * 1000.0
    );
    Ok(())
}

fn build_deltas(count: usize, pages: u64) -> Vec<Delta> {
    (0..count)
        .map(|idx| {
            let page_id = PageID(((idx as u64) % pages) + 1);
            let payload = vec![(idx % 255) as u8; 128];
            Delta {
                delta_id: DeltaID(idx as u64),
                page_id,
                epoch: Epoch(idx as u32),
                mask: vec![true; payload.len()],
                payload,
                is_sparse: false,
                timestamp: idx as u64,
                source: Source("phase6_bench".into()),
                intent_metadata: None,
            }
        })
        .collect()
}

fn build_graph(nodes: u64) -> ShadowPageGraph {
    let graph = ShadowPageGraph::default();
    for id in 1..nodes {
        graph.add_edge(PageID(id), PageID(id + 1), EdgeType::Data);
    }
    graph
}

fn write_report(
    throughput: &ThroughputMetrics,
    tick: &TickMetrics,
    workers: usize,
) -> Result<(), Box<dyn Error>> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_secs_f64();
    let mut file = File::create("benchmark/results/phase6.json")?;
    writeln!(
        file,
        r#"{{
  "timestamp": "{timestamp}",
  "throughput": {{
    "processed": {processed},
    "duration_us": {duration_us},
    "throughput_per_sec_single": {throughput_sec},
    "throughput_per_sec_multi": {throughput_multi}
  }},
  "tick_latency": {{
    "propagation_us": {prop_us},
    "graph_us": {graph_us},
    "gc_us": {gc_us},
    "total_us": {total_us},
    "meets_budget_ms": {meets_budget},
    "gc_invoked": {gc_invoked},
    "graph_has_cycle": {graph_cycle}
  }}
}}"#,
        processed = throughput.processed,
        duration_us = throughput.duration.as_micros(),
        throughput_sec = throughput.throughput,
        throughput_multi = throughput.throughput * workers as f64,
        prop_us = tick.propagation.as_micros(),
        graph_us = tick.graph_validation.as_micros(),
        gc_us = tick.gc.as_micros(),
        total_us = tick.total.as_micros(),
        meets_budget = (tick.total.as_secs_f64() * 1000.0) < 16.0,
        gc_invoked = tick.gc_invoked,
        graph_cycle = tick.graph_has_cycle,
    )?;
    Ok(())
}
