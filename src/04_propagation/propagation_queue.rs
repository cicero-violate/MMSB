use super::propagation_command_buffer::PropagationCommand;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Default)]
pub struct PropagationQueue {
    queue: Mutex<VecDeque<PropagationCommand>>,
    has_work: Arc<AtomicBool>,
}

impl PropagationQueue {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            has_work: Arc::new(AtomicBool::new(false)),
        }
    }
    
    pub fn push(&self, command: PropagationCommand) {
        self.queue.lock().push_back(command);
        self.has_work.store(true, Ordering::Release);
    }

    pub fn pop(&self) -> Option<PropagationCommand> {
        let result = self.queue.lock().pop_front();
        if result.is_none() {
            self.has_work.store(false, Ordering::Release);
        }
        result
    }
    
    pub fn push_batch(&self, commands: Vec<PropagationCommand>) {
        let mut queue = self.queue.lock();
        queue.extend(commands);
        self.has_work.store(!queue.is_empty(), Ordering::Release);
    }
    
    pub fn drain_batch(&self, max_count: usize) -> Vec<PropagationCommand> {
        let mut queue = self.queue.lock();
        let count = queue.len().min(max_count);
        let batch: Vec<_> = queue.drain(..count).collect();
        self.has_work.store(!queue.is_empty(), Ordering::Release);
        batch
    }
    
    pub fn is_empty(&self) -> bool {
        !self.has_work.load(Ordering::Acquire)
    }
    
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }
}
