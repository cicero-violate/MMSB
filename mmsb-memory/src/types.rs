//! Centralized type definitions for mmsb-memory

//! Pure data types only - no error handling frameworks

// Core identifiers
pub use crate::page::page_types::{PageID, PageError, PageLocation};
pub use crate::epoch::epoch_types::Epoch;
pub use crate::dag::edge_types::EdgeType;

// Delta types
pub use crate::delta::delta_types::{DeltaID, DeltaError, Source};
