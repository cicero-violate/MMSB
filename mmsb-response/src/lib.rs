//! MMSB Response Module

use mmsb_events::{MMSBSubscription, StateQuery, StateProjection, Delta};

pub struct ResponseModule;

impl ResponseModule {
    pub fn new() -> Self {
        Self
    }
    
    pub fn query(&self, query: StateQuery) -> StateProjection {
        StateProjection { data: vec![] }
    }
}

impl Default for ResponseModule {
    fn default() -> Self {
        Self::new()
    }
}
