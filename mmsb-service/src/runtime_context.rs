use std::sync::{Arc, Mutex};

use crate::ProtocolSignalIn;
use crate::Scheduler;
use mmsb_events::{StateBus, ExecutionBus, LearningBus, JudgmentBus};

pub struct RuntimeContext {
    scheduler: Arc<dyn Scheduler + Send>,
    state_bus: Arc<Mutex<dyn StateBus + Send>>,
    execution_bus: Arc<Mutex<dyn ExecutionBus + Send>>,
    learning_bus: Arc<Mutex<dyn LearningBus + Send>>,
    protocol_in: ProtocolSignalIn,
    judgment_bus: Option<Arc<Mutex<dyn JudgmentBus + Send>>>,
}

impl RuntimeContext {
    pub fn new(
        scheduler: Arc<dyn Scheduler + Send>,
        state_bus: Arc<Mutex<dyn StateBus + Send>>,
        execution_bus: Arc<Mutex<dyn ExecutionBus + Send>>,
        learning_bus: Arc<Mutex<dyn LearningBus + Send>>,
        protocol_in: ProtocolSignalIn,
        judgment_bus: Option<Arc<Mutex<dyn JudgmentBus + Send>>>,
    ) -> Self {
        Self {
            scheduler,
            state_bus,
            execution_bus,
            learning_bus,
            protocol_in,
            judgment_bus,
        }
    }

    pub fn protocol_in(&self) -> &ProtocolSignalIn {
        &self.protocol_in
    }
}

use crate::ServiceContext;

impl ServiceContext for RuntimeContext {
    fn with_state_bus<F>(&self, f: F) where F: FnOnce(&mut dyn StateBus) {
        let mut sb = self.state_bus.lock().unwrap();
        f(&mut *sb);
    }

    fn with_execution_bus<F>(&self, f: F) where F: FnOnce(&mut dyn ExecutionBus) {
        let mut eb = self.execution_bus.lock().unwrap();
        f(&mut *eb);
    }

    fn with_learning_bus<F>(&self, f: F) where F: FnOnce(&mut dyn LearningBus) {
        let mut lb = self.learning_bus.lock().unwrap();
        f(&mut *lb);
    }

    fn with_judgment_bus<F>(&self, f: F) -> bool where F: FnOnce(&mut dyn JudgmentBus) {
        if let Some(jb) = &self.judgment_bus {
            let mut jb = jb.lock().unwrap();
            f(&mut *jb);
            true
        } else {
            false
        }
    }
}
