//! Commit Notifier - Event Emission for Memory Commits
//!
//! This is NOT a service - it's a communication primitive.
//! MemoryEngine emits events through this notifier when commits occur.
//! Services subscribe to receive work notifications instantly (zero polling).

use tokio::sync::broadcast;
use mmsb_events::MemoryCommitted;

/// CommitNotifier - broadcasts MemoryCommitted events
///
/// This is infrastructure, not a worker:
/// - Created once at startup
/// - Injected into MemoryEngine
/// - Services subscribe via MemoryReader
#[derive(Clone)]
pub struct CommitNotifier {
    tx: broadcast::Sender<MemoryCommitted>,
}

impl CommitNotifier {
    /// Create a new notifier with specified channel capacity
    ///
    /// Returns (notifier, initial_receiver)
    /// The initial receiver prevents "no receiver" errors
    pub fn new(capacity: usize) -> (Self, broadcast::Receiver<MemoryCommitted>) {
        let (tx, rx) = broadcast::channel(capacity);
        (Self { tx }, rx)
    }
    
    /// Emit a commit event (non-blocking)
    ///
    /// If a receiver is lagging and the channel is full, the oldest
    /// message is dropped (lagged policy). This prevents slow services
    /// from blocking memory commits.
    pub fn notify(&self, event: MemoryCommitted) {
        // Don't panic if all receivers dropped - just log and continue
        if let Err(_) = self.tx.send(event) {
            // All receivers dropped, event is lost
            // In production, might want to log this
        }
    }
    
    /// Create a new subscription to commit events
    ///
    /// Services call this to receive notifications.
    /// Each subscriber gets their own Receiver.
    pub fn subscribe(&self) -> broadcast::Receiver<MemoryCommitted> {
        self.tx.subscribe()
    }
    
    /// Get number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }
}

impl std::fmt::Debug for CommitNotifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitNotifier")
            .field("subscribers", &self.subscriber_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmsb_primitives::{EventId, Timestamp, Hash, PageID};
    use mmsb_proof::{CommitProof, AdmissionProof, OutcomeProof};

    #[tokio::test]
    async fn test_notifier_basic() {
        let (notifier, _rx) = CommitNotifier::new(10);
        
        // Create a subscriber
        let mut sub = notifier.subscribe();
        
        // Emit an event
        let event = MemoryCommitted {
            event_id: EventId(1),
            timestamp: Timestamp(100),
            commit_proof: CommitProof {
                admission_proof_hash: Hash::default(),
                delta_hash: Hash::default(),
                state_hash: Hash::default(),
                invariants_held: true,
            },
            delta_hash: Hash::default(),
            epoch: 1,
            snapshot_ref: Hash::default(),
            admission_proof: AdmissionProof {
                judgment_proof_hash: Hash::default(),
                epoch: 1,
                nonce: 0,
            },
            outcome_proof: OutcomeProof {
                commit_proof_hash: Hash::default(),
                success: true,
                error_class: None,
                rollback_hash: None,
            },
            affected_page_ids: vec![PageID(0)],
        };
        
        notifier.notify(event.clone());
        
        // Receive the event
        let received = sub.recv().await.unwrap();
        assert_eq!(received.epoch, 1);
    }
    
    #[tokio::test]
    async fn test_multiple_subscribers() {
        let (notifier, _rx) = CommitNotifier::new(10);
        
        let mut sub1 = notifier.subscribe();
        let mut sub2 = notifier.subscribe();
        
        assert_eq!(notifier.subscriber_count(), 2);
        
        // Emit event - both should receive
        let event = MemoryCommitted {
            event_id: EventId(1),
            timestamp: Timestamp(100),
            commit_proof: CommitProof {
                admission_proof_hash: Hash::default(),
                delta_hash: Hash::default(),
                state_hash: Hash::default(),
                invariants_held: true,
            },
            delta_hash: Hash::default(),
            epoch: 1,
            snapshot_ref: Hash::default(),
            admission_proof: AdmissionProof {
                judgment_proof_hash: Hash::default(),
                epoch: 1,
                nonce: 0,
            },
            outcome_proof: OutcomeProof {
                commit_proof_hash: Hash::default(),
                success: true,
                error_class: None,
                rollback_hash: None,
            },
            affected_page_ids: vec![],
        };
        
        notifier.notify(event);
        
        let r1 = sub1.recv().await.unwrap();
        let r2 = sub2.recv().await.unwrap();
        
        assert_eq!(r1.epoch, 1);
        assert_eq!(r2.epoch, 1);
    }
}
