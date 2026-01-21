use mmsb_events::{StateBus, ExecutionBus, LearningBus, JudgmentBus};

pub trait ServiceContext: Send + Sync {
    fn with_state_bus<F>(&self, f: F) where F: FnOnce(&mut dyn StateBus);
    fn with_execution_bus<F>(&self, f: F) where F: FnOnce(&mut dyn ExecutionBus);
    fn with_learning_bus<F>(&self, f: F) where F: FnOnce(&mut dyn LearningBus);
    fn with_judgment_bus<F>(&self, f: F) -> bool where F: FnOnce(&mut dyn JudgmentBus);
}
