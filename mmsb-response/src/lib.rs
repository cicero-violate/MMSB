//! MMSB Response Module
//! Formats agent-facing responses from MMSB subscriptions

use mmsb_events::{MMSBSubscription, StateQuery, StateProjection, Delta, AnyEvent};

pub struct ResponseModule {
    subscription: Option<Box<dyn MMSBSubscription>>,
}

impl ResponseModule {
    pub fn new() -> Self {
        Self { subscription: None }
    }
    
    pub fn attach_subscription(&mut self, sub: Box<dyn MMSBSubscription>) {
        self.subscription = Some(sub);
    }
    
    pub fn query(&self, query: StateQuery) -> StateProjection {
        self.subscription
            .as_ref()
            .map(|s| s.project_view(query))
            .unwrap_or_else(|| StateProjection)
    }
}

impl Default for ResponseModule {
    fn default() -> Self {
        Self::new()
    }
}
