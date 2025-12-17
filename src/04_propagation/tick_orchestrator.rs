use super::throughput_engine::ThroughputEngine;
use crate::dag::{GraphValidator, ShadowPageGraph};
use crate::page::{Delta, PageError};
use crate::utility::MemoryMonitor;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TickMetrics {
    pub propagation: Duration,
    pub graph_validation: Duration,
    pub gc: Duration,
    pub total: Duration,
    pub throughput: f64,
    pub gc_invoked: bool,
    pub graph_has_cycle: bool,
    pub processed: usize,
}

pub struct TickOrchestrator {
    throughput: ThroughputEngine,
    graph: Arc<ShadowPageGraph>,
    memory_monitor: Arc<MemoryMonitor>,
    tick_budget_ms: u64,
}

impl TickOrchestrator {
    pub fn new(
        throughput: ThroughputEngine,
        graph: Arc<ShadowPageGraph>,
        memory_monitor: Arc<MemoryMonitor>,
    ) -> Self {
        Self {
            throughput,
            graph,
            memory_monitor,
            tick_budget_ms: 16,
        }
    }

    pub fn execute_tick(&self, deltas: Vec<Delta>) -> Result<TickMetrics, PageError> {
        let tick_start = Instant::now();
        let throughput_metrics = self.throughput.process_parallel(deltas)?;

        let graph_report = {
            let validator = GraphValidator::new(&self.graph);
            validator.detect_cycles()
        };

        let gc_metrics = self
            .memory_monitor
            .trigger_incremental_gc(self.memory_monitor.config().incremental_batch_pages);

        let total = tick_start.elapsed();
        Ok(TickMetrics {
            propagation: throughput_metrics.duration,
            graph_validation: graph_report.duration,
            gc: gc_metrics.map(|m| m.duration).unwrap_or_default(),
            total,
            throughput: throughput_metrics.throughput,
            gc_invoked: gc_metrics.is_some(),
            graph_has_cycle: graph_report.has_cycle,
            processed: throughput_metrics.processed,
        })
    }

    pub fn budget_ms(&self) -> u64 {
        self.tick_budget_ms
    }
}

#[cfg(test)]
mod tests {
    use super::TickOrchestrator;
    use crate::dag::{EdgeType, ShadowPageGraph};
    use crate::page::{Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source};
    use crate::types::Epoch;
    use crate::utility::{MemoryMonitor, MemoryMonitorConfig};
    use super::ThroughputEngine;
    use std::sync::Arc;

    fn sample_delta(id: u64, page: u64, value: u8) -> Delta {
        Delta {
            delta_id: DeltaID(id),
            page_id: PageID(page),
            epoch: Epoch(id as u32),
            mask: vec![true; 32],
            payload: vec![value; 32],
            is_sparse: false,
            timestamp: id,
            source: Source(format!("delta-{id}")),
            intent_metadata: None,
        }
    }

    fn orchestrator(threshold: usize) -> (TickOrchestrator, Arc<PageAllocator>) {
        let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
        for id in 1..=4 {
            allocator
                .allocate_raw(PageID(id), 32, Some(PageLocation::Cpu))
                .unwrap();
        }
        let throughput = ThroughputEngine::new(Arc::clone(&allocator), 2, 64);
        let graph = Arc::new(ShadowPageGraph::default());
        graph.add_edge(PageID(1), PageID(2), EdgeType::Data);
        let cold_limit = if threshold == usize::MAX { u64::MAX } else { 0 };
        let memory = Arc::new(MemoryMonitor::with_config(
            Arc::clone(&allocator),
            MemoryMonitorConfig {
                gc_threshold_bytes: threshold,
                cold_page_age_limit: cold_limit,
                ..MemoryMonitorConfig::default()
            },
        ));
        (
            TickOrchestrator::new(throughput, graph, memory),
            allocator,
        )
    }

    #[test]
    fn tick_metrics_capture_all_phases() {
        let (orchestrator, _) = orchestrator(usize::MAX);
        let deltas = vec![sample_delta(1, 1, 0xAA); 64];
        let metrics = orchestrator.execute_tick(deltas).unwrap();
        assert!(metrics.total >= metrics.propagation);
        assert!(!metrics.gc_invoked);
        assert_eq!(metrics.processed, 64);
    }

    #[test]
    fn gc_invoked_when_threshold_low() {
        let (orchestrator, _) = orchestrator(1);
        let deltas = vec![sample_delta(1, 1, 0xAA); 8];
        let metrics = orchestrator.execute_tick(deltas).unwrap();
        assert!(metrics.gc_invoked);
    }
}
