//! MMSB Chromium Module
//! Browser automation - web interaction

use mmsb_events::{ChromiumProtocol, BrowserCommand, BrowserResult, MMSBSubscription};

pub struct ChromiumModule {
    subscription: Option<Box<dyn MMSBSubscription>>,
}

impl ChromiumModule {
    pub fn new() -> Self {
        Self { subscription: None }
    }
    
    pub fn attach_subscription(&mut self, sub: Box<dyn MMSBSubscription>) {
        self.subscription = Some(sub);
    }
}

impl ChromiumProtocol for ChromiumModule {
    fn execute_command(&mut self, _command: BrowserCommand) -> BrowserResult {
        // Stub: browser automation would happen here
        BrowserResult
    }
    
    fn report_observation(&mut self, _result: BrowserResult) {
        // Stub: write to StateBus
    }
}

impl Default for ChromiumModule {
    fn default() -> Self {
        Self::new()
    }
}
