use super::throughput_engine::ThroughputEngine;
use crate::dag::{GraphValidator, ShadowPageGraph, DependencyGraph};
use mmsb_judgment::JudgmentToken;
use crate::page::{commit_delta, Delta, PageError, TransactionLog};
use crate::utility::{MmsbAdmissionProof, MmsbExecutionProof};
use crate::types::{MemoryPressureHandler};
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
    dag: Arc<DependencyGraph>,
    memory_monitor: Arc<dyn MemoryPressureHandler>,
    tick_budget_ms: u64,
}

impl TickOrchestrator {
    pub fn new(
        throughput: ThroughputEngine,
        graph: Arc<ShadowPageGraph>,
        dag: Arc<DependencyGraph>,
        memory_monitor: Arc<dyn MemoryPressureHandler>,
    ) -> Self {
        Self {
            throughput,
            graph,
            dag,
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
            .run_gc(self.memory_monitor.incremental_batch_pages());

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

pub(crate) fn request_commit(
    log: &TransactionLog,
    token: &JudgmentToken,
    admission_proof: &MmsbAdmissionProof,
    execution_proof: &MmsbExecutionProof,
    delta: Delta,
    dag: Option<&DependencyGraph>,
) -> std::io::Result<()> {
    commit_delta(log, token, admission_proof, execution_proof, delta, dag)
}

pub(crate) fn submit_intent(
    _log: &TransactionLog,
    _delta: Delta,
) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod judgment_commit_test {
    use super::request_commit;
    use mmsb_judgment::JudgmentToken;
    use crate::page::{tlog, Delta, DeltaID, Epoch, PageID, Source, TransactionLog};
    use crate::utility::{ADMISSION_PROOF_VERSION, EXECUTION_PROOF_VERSION, MmsbAdmissionProof, MmsbExecutionProof};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn human_judgment_commit_once() -> std::io::Result<()> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mmsb_judgment_commit_{nanos}.tlog"));
        let log = TransactionLog::new(path)?;

        let delta = Delta {
            delta_id: DeltaID(1),
            page_id: PageID(1),
            epoch: Epoch(1),
            mask: vec![true; 4],
            payload: vec![0xAB; 4],
            is_sparse: false,
            timestamp: 1,
            source: Source("human-test".to_string()),
            intent_metadata: Some("human-approved commit".to_string()),
        };

        let delta_hash = tlog::delta_hash(&delta);
        let admission_proof = MmsbAdmissionProof {
            version: ADMISSION_PROOF_VERSION,
            delta_hash: delta_hash.clone(),
            conversation_id: "test".to_string(),
            message_id: "test".to_string(),
            suffix: "0".to_string(),
            intent_hash: "test".to_string(),
            approved: true,
            command: Vec::new(),
            cwd: None,
            env: None,
            epoch: 0,
        };
        let execution_proof = MmsbExecutionProof {
            version: EXECUTION_PROOF_VERSION,
            delta_hash,
            tool_call_id: "test".to_string(),
            tool_name: "test".to_string(),
            output: serde_json::json!({}),
            epoch: 0,
        };
        let token = JudgmentToken::test_only();
        request_commit(&log, &token, &admission_proof, &execution_proof, delta, None)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::TickOrchestrator;
    use crate::dag::{EdgeType, ShadowPageGraph};
    use crate::page::{Delta, DeltaID, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source};
    use crate::types::{Epoch, GCMetrics, MemoryPressureHandler};
    use super::ThroughputEngine;
    use std::sync::Arc;
    use std::time::Duration;

    struct TestMemoryHandler {
        batch: usize,
        collect: bool,
    }

    impl TestMemoryHandler {
        fn new(batch: usize, collect: bool) -> Self {
            Self { batch, collect }
        }
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
        let memory: Arc<dyn MemoryPressureHandler> =
            Arc::new(TestMemoryHandler::new(32, threshold != usize::MAX));
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
