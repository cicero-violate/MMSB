use super::propagation_command_buffer::PropagationCommand;
use parking_lot::Mutex;
use std::collections::VecDeque;

#[derive(Debug, Default)]
pub struct PropagationQueue {
    queue: Mutex<VecDeque<PropagationCommand>>,
}

impl PropagationQueue {
    pub fn push(&self, command: PropagationCommand) {
        self.queue.lock().push_back(command);
    }

    pub fn pop(&self) -> Option<PropagationCommand> {
        self.queue.lock().pop_front()
    }
}
