use std::sync::Arc;
use parking_lot::Mutex;

use crate::ProtocolSignalIn;
use crate::Scheduler;
use mmsb_events::{StateBus, ExecutionBus, LearningBus, JudgmentBus, MemoryReader};

pub struct RuntimeContext {
    scheduler: Arc<dyn Scheduler + Send>,
    state_bus: Arc<Mutex<dyn StateBus + Send>>,
    memory_reader: Arc<dyn MemoryReader + Send + Sync>,
    execution_bus: Arc<Mutex<dyn ExecutionBus + Send>>,
    learning_bus: Arc<Mutex<dyn LearningBus + Send>>,
    protocol_in: ProtocolSignalIn,
    judgment_bus: Option<Arc<Mutex<dyn JudgmentBus + Send>>>,
}

impl RuntimeContext {
    pub fn new(
        scheduler: Arc<dyn Scheduler + Send>,
        state_bus: Arc<Mutex<dyn StateBus + Send>>,
        memory_reader: Arc<dyn MemoryReader + Send + Sync>,
        execution_bus: Arc<Mutex<dyn ExecutionBus + Send>>,
        learning_bus: Arc<Mutex<dyn LearningBus + Send>>,
        protocol_in: ProtocolSignalIn,
        judgment_bus: Option<Arc<Mutex<dyn JudgmentBus + Send>>>,
    ) -> Self {
        Self {
            scheduler,
            state_bus,
            memory_reader,
            execution_bus,
            learning_bus,
            protocol_in,
            judgment_bus,
        }
    }

    pub fn protocol_in(&self) -> &ProtocolSignalIn {
        &self.protocol_in
    }
    
    /// Get read-only memory access
    pub fn memory_reader(&self) -> &Arc<dyn MemoryReader + Send + Sync> {
        &self.memory_reader
    }
}

use crate::ServiceContext;

impl ServiceContext for RuntimeContext {
    fn with_state_bus<F>(&self, f: F) where F: FnOnce(&mut dyn StateBus) {
        let mut sb = self.state_bus.lock();
        f(&mut *sb);
    }

    fn with_execution_bus<F>(&self, f: F) where F: FnOnce(&mut dyn ExecutionBus) {
        let mut eb = self.execution_bus.lock();
        f(&mut *eb);
    }

    fn with_learning_bus<F>(&self, f: F) where F: FnOnce(&mut dyn LearningBus) {
        let mut lb = self.learning_bus.lock();
        f(&mut *lb);
    }

    fn with_judgment_bus<F>(&self, f: F) -> bool where F: FnOnce(&mut dyn JudgmentBus) {
        if let Some(jb) = &self.judgment_bus {
            let mut jb = jb.lock();
            f(&mut *jb);
            true
        } else {
            false
        }
    }
}
