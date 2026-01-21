//! Notifier Infrastructure
//!
//! Event emission components for memory operations.
//! These are NOT services - they're communication primitives.

pub mod commit_notifier;

pub use commit_notifier::CommitNotifier;