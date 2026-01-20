//! MMSB Compute Module
//! GPU/CUDA acceleration for parallel computation

use mmsb_events::{ComputeProtocol, ComputeRequest, ComputeResult, MMSBSubscription};

pub struct ComputeModule {
    subscription: Option<Box<dyn MMSBSubscription>>,
}

impl ComputeModule {
    pub fn new() -> Self {
        Self { subscription: None }
    }
    
    pub fn attach_subscription(&mut self, sub: Box<dyn MMSBSubscription>) {
        self.subscription = Some(sub);
    }
}

impl ComputeProtocol for ComputeModule {
    fn compute(&mut self, request: ComputeRequest) -> ComputeResult {
        // Stub: GPU computation would happen here
        ComputeResult
    }
    
    fn report_result(&mut self, _result: ComputeResult) {
        // Stub: write to StateBus
    }
}

impl Default for ComputeModule {
    fn default() -> Self {
        Self::new()
    }
}
