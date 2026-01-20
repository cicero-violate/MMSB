//! MMSBSubscription - Read-only projection from MMSB services --> MMSB-memory

use crate::state_bus::Delta;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateQuery {
    pub query_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateProjection {
    pub data: Vec<u8>,
}

pub trait MMSBSubscription {
    fn subscribe_deltas(&mut self) -> Box<dyn Iterator<Item = Delta>>;
    fn project_view(&self, query: StateQuery) -> StateProjection;
}
