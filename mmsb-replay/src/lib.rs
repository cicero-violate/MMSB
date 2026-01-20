//! MMSB Replay Module

use mmsb_events::{MMSBSubscription, StateQuery, StateProjection, Delta};

pub struct ReplayModule;

impl ReplayModule {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReplayModule {
    fn default() -> Self {
        Self::new()
    }
}

impl MMSBSubscription for ReplayModule {
    fn subscribe_deltas(&mut self) -> Box<dyn Iterator<Item = Delta>> {
        Box::new(std::iter::empty())
    }
    
    fn project_view(&self, _query: StateQuery) -> StateProjection {
        StateProjection { data: vec![] }
    }
}
