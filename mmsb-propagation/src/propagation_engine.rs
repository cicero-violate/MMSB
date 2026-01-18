use super::propagation_command_buffer::PropagationCommand;
use super::propagation_queue::PropagationQueue;

// Placeholder - awaiting full integration with mmsb-memory
pub struct DependencyGraph;
impl DependencyGraph {
    pub fn version(&self) -> u64 { 0 }
}

pub struct PropagationEngine {
    queue: PropagationQueue,
}

impl Default for PropagationEngine {
    fn default() -> Self {
        Self {
            queue: PropagationQueue::default(),
        }
    }
}

impl PropagationEngine {
    pub fn enqueue(&self, command: PropagationCommand) {
        self.queue.push(command);
    }

    pub fn drain(&self, _dag: &DependencyGraph) {
        // Stub - awaiting full integration
    }
}
