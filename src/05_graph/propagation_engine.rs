use super::propagation_command_buffer::PropagationCommand;
use super::propagation_queue::PropagationQueue;
use crate::types::{Page, PageID};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

type Callback = Arc<dyn Fn(&Page, &[Arc<Page>]) + Send + Sync>;

pub struct PropagationEngine {
    callbacks: RwLock<HashMap<PageID, Callback>>,
    queue: PropagationQueue,
}

impl Default for PropagationEngine {
    fn default() -> Self {
        Self {
            callbacks: RwLock::new(HashMap::new()),
            queue: PropagationQueue::default(),
        }
    }
}

impl PropagationEngine {
    pub fn register_callback(&self, page_id: PageID, callback: Callback) {
        self.callbacks.write().insert(page_id, callback);
    }

    pub fn enqueue(&self, command: PropagationCommand) {
        self.queue.push(command);
    }

    pub fn drain(&self) {
        while let Some(command) = self.queue.pop() {
            if let Some(cb) = self.callbacks.read().get(&command.page_id) {
                (*cb)(&command.page, &command.dependencies);
            }
        }
    }
}
