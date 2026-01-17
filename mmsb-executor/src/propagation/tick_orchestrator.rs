use super::throughput_engine::ThroughputEngine;
use mmsb_events::{Event, MemoryCommitted};  // Import from mmsb-events crate
use std::sync::Arc;
use std::time::{Duration, Instant};

// Assuming MemoryPressureHandler is a substrate-only trait (hardware monitoring).
// If it's semantic, move it to mmsb-memory and expose via events.
pub trait MemoryPressureHandler: Send + Sync {
    fn incremental_batch_pages(&self) -> usize;
    fn run_gc(&self, budget_pages: usize) -> Option<GCMetrics>;
}

// Minimal GCMetrics (keep if needed; otherwise derive from events)
#[derive(Debug, Clone)]
pub struct GCMetrics {
    pub reclaimed_pages: usize,
    pub reclaimed_bytes: usize,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct TickMetrics {
    pub propagation: Duration,
    pub gc: Duration,
    pub total: Duration,
    pub throughput: f64,
    pub gc_invoked: bool,
    pub processed: usize,  // Now: number of affected page IDs from event
}

pub struct TickOrchestrator {
    throughput: ThroughputEngine,
    memory_monitor: Arc<dyn MemoryPressureHandler>,
    tick_budget_ms: u64,
}

impl TickOrchestrator {
    pub fn new(
        throughput: ThroughputEngine,
        memory_monitor: Arc<dyn MemoryPressureHandler>,
    ) -> Self {
        Self {
            throughput,
            memory_monitor,
            tick_budget_ms: 16,
        }
    }

    // Now event-driven: Reacts to MemoryCommitted event from the bus.
    // Processes only affected page IDs (physical propagation, no semantic logic).
    pub fn execute_tick(&self, event: &MemoryCommitted) -> Result<TickMetrics, PageError> {
        let tick_start = Instant::now();

        // Extract minimal payload from event (no full Deltas or graphs)
        let affected_pages = &event.affected_page_ids;  // Assuming we add this field to MemoryCommitted (see below)

        // Physical propagation only (best-effort hardware scheduling)
        let throughput_metrics = self.throughput.process_parallel(affected_pages)?;

        // No graph validation or commit – that's memory's job

        // GC is substrate (hardware pressure), so keep it
        let gc_metrics = self
            .memory_monitor
            .run_gc(self.memory_monitor.incremental_batch_pages());

        let total = tick_start.elapsed();
        Ok(TickMetrics {
            propagation: throughput_metrics.duration,
            gc: gc_metrics.map(|m| m.duration).unwrap_or_default(),
            total,
            throughput: throughput_metrics.throughput,
            gc_invoked: gc_metrics.is_some(),
            processed: affected_pages.len(),
        })
    }

    pub fn budget_ms(&self) -> u64 {
        self.tick_budget_ms
    }
}

#[cfg(test)]
mod tests {
    use super::{GCMetrics, MemoryPressureHandler, TickOrchestrator, TickMetrics};
    use mmsb_events::MemoryCommitted;  // Use real event struct
    use mmsb_proof::Hash;  // Only hashes/IDs allowed
    use std::sync::Arc;
    use std::time::Duration;

    // Mock impl for testing (hardware-only)
    struct TestMemoryHandler {
        batch: usize,
        collect: bool,
    }

    impl MemoryPressureHandler for TestMemoryHandler {
        fn incremental_batch_pages(&self) -> usize {
            self.batch
        }

        fn run_gc(&self, _budget_pages: usize) -> Option<GCMetrics> {
            if !self.collect {
                return None;
            }
            Some(GCMetrics {
                reclaimed_pages: self.batch,
                reclaimed_bytes: self.batch * 4096,
                duration: Duration::from_millis(1),
            })
        }
    }

    // Helper to create mock MemoryCommitted event
    fn mock_event(affected_count: usize) -> MemoryCommitted {
        MemoryCommitted {
            event_id: Hash::default(),  // Dummy hash
            timestamp: 0,
            delta_hash: Hash::default(),
            epoch: 1,
            snapshot_ref: None,
            admission_proof: Default::default(),  // Minimal or mocked
            commit_proof: Default::default(),
            outcome_proof: Default::default(),
            affected_page_ids: (1..=affected_count).map(|id| PageID(id as u64)).collect(),  // Assuming PageID is a simple type; expose minimally from mmsb-memory if needed
        }
    }

    fn orchestrator(threshold: usize) -> (TickOrchestrator, Arc<PageAllocator>) {  // Assuming PageAllocator is substrate-only
        // ... (keep similar, but remove DAG/ops – no graph here)
        let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
        let throughput = ThroughputEngine::new(Arc::clone(&allocator), 2, 64);
        let memory: Arc<dyn MemoryPressureHandler> =
            Arc::new(TestMemoryHandler::new(32, threshold != usize::MAX));
        (
            TickOrchestrator::new(throughput, memory),
            allocator,
        )
    }

    #[test]
    fn tick_metrics_capture_all_phases() {
        let (orchestrator, _) = orchestrator(usize::MAX);
        let event = mock_event(64);  // Simulate event with 64 affected pages
        let metrics = orchestrator.execute_tick(&event).unwrap();
        assert!(metrics.total >= metrics.propagation);
        assert!(!metrics.gc_invoked);
        assert_eq!(metrics.processed, 64);
    }

    #[test]
    fn gc_invoked_when_threshold_low() {
        let (orchestrator, _) = orchestrator(1);
        let event = mock_event(8);  // Simulate event with 8 affected pages
        let metrics = orchestrator.execute_tick(&event).unwrap();
        assert!(metrics.gc_invoked);
    }
}
