use super::propagation_command_buffer::PropagationCommand;
use super::ring_buffer::LockFreeRingBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

const DEFAULT_CAPACITY: usize = 1 << 13; // 8192 entries

pub struct PropagationQueue {
    ring: LockFreeRingBuffer<PropagationCommand>,
    has_work: AtomicBool,
}

impl Default for PropagationQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PropagationQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropagationQueue")
            .field("len", &self.len())
            .field("capacity", &self.ring.capacity())
            .finish()
    }
}

impl PropagationQueue {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ring: LockFreeRingBuffer::new(capacity),
            has_work: AtomicBool::new(false),
        }
    }

    pub fn push(&self, command: PropagationCommand) {
        self.push_internal(command);
    }

    pub fn pop(&self) -> Option<PropagationCommand> {
        match self.ring.try_pop() {
            Some(cmd) => Some(cmd),
            None => {
                self.has_work.store(false, Ordering::Release);
                None
            }
        }
    }

    pub fn push_batch(&self, commands: Vec<PropagationCommand>) {
        for command in commands {
            self.push_internal(command);
        }
    }

    pub fn drain_batch(&self, max_count: usize) -> Vec<PropagationCommand> {
        if max_count == 0 {
            return Vec::new();
        }
        let drained = self.ring.pop_batch(max_count);
        if drained.is_empty() {
            self.has_work.store(false, Ordering::Release);
        }
        drained
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    fn push_internal(&self, mut command: PropagationCommand) {
        loop {
            match self.ring.try_push(command) {
                Ok(()) => {
                    self.has_work.store(true, Ordering::Release);
                    break;
                }
                Err(cmd) => {
                    command = cmd;
                    thread::yield_now();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{Page, PageID, PageLocation};
    use std::sync::Arc;

    fn command(id: u64) -> PropagationCommand {
        let page = Arc::new(Page::new(PageID(id), 8, PageLocation::Cpu).unwrap());
        PropagationCommand {
            page_id: page.id,
            page,
            dependencies: Vec::new(),
        }
    }

    #[test]
    fn queue_roundtrip() {
        let queue = PropagationQueue::with_capacity(8);
        for i in 0..8 {
            queue.push(command(i));
        }
        assert_eq!(queue.len(), 8);
        for i in 0..8 {
            let popped = queue.pop().unwrap();
            assert_eq!(popped.page_id, PageID(i));
        }
        assert!(queue.pop().is_none());
    }

    #[test]
    fn drain_batch_respects_bounds() {
        let queue = PropagationQueue::with_capacity(8);
        for i in 0..6 {
            queue.push(command(i));
        }
        let drained = queue.drain_batch(4);
        assert_eq!(drained.len(), 4);
        assert_eq!(queue.len(), 2);
    }
}
