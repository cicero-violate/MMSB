use super::propagation_command_buffer::PropagationCommand;
use super::propagation_queue::PropagationQueue;

/// Minimal placeholder for sparse message passing.
pub fn enqueue_sparse(queue: &PropagationQueue, command: PropagationCommand) {
    queue.push(command);
}
