// src/wake_source.rs
use crate::Scheduler;

/// Source of wake events.
/// Bridges external truth edges (e.g. memory commits) into the scheduler.
pub trait WakeSource: Send + Sync {
    /// Attach this wake source to a scheduler.
    /// Implementations call `scheduler.wake()` when triggered.
    fn attach(&self, scheduler: &dyn Scheduler);
}
