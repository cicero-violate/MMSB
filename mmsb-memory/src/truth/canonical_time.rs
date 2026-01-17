//! Canonical Time Management
//! 
//! Memory owns canonical time and epoch ordering.

#[derive(Debug, Clone)]
pub struct CanonicalTime {
    logical_time: u64,
    current_epoch: u64,
}

impl CanonicalTime {
    pub fn new() -> Self {
        Self {
            logical_time: 0,
            current_epoch: 0,
        }
    }

    pub fn next(&mut self) -> u64 {
        self.logical_time += 1;
        self.logical_time
    }

    pub fn current(&self) -> u64 {
        self.logical_time
    }

    pub fn epoch(&self) -> u64 {
        self.current_epoch
    }

    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }
}

impl Default for CanonicalTime {
    fn default() -> Self {
        Self::new()
    }
}
