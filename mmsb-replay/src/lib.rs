//! MMSB Replay Module
//! Historical event stream for replay and audit (read-only)

use mmsb_events::{ReplayProtocol, MMSBSubscription, EventStream, StateSnapshot, Delta, AnyEvent, StateQuery, StateProjection};

pub struct ReplayModule {
    subscription: Option<Box<dyn MMSBSubscription>>,
}

impl ReplayModule {
    pub fn new() -> Self {
        Self { subscription: None }
    }
    
    pub fn attach_subscription(&mut self, sub: Box<dyn MMSBSubscription>) {
        self.subscription = Some(sub);
    }
}

impl MMSBSubscription for ReplayModule {
    fn subscribe_deltas(&mut self) -> Box<dyn Iterator<Item = Delta>> {
        Box::new(std::iter::empty())
    }
    
    fn subscribe_events(&mut self) -> Box<dyn Iterator<Item = AnyEvent>> {
        Box::new(std::iter::empty())
    }
    
    fn project_view(&self, _query: StateQuery) -> StateProjection {
        StateProjection
    }
}

impl ReplayProtocol for ReplayModule {
    fn stream_events(&self, _from_epoch: u64, _to_epoch: u64) -> EventStream {
        EventStream
    }
    
    fn replay_to_state(&mut self, _target_epoch: u64) -> StateSnapshot {
        StateSnapshot
    }
}

impl Default for ReplayModule {
    fn default() -> Self {
        Self::new()
    }
}
