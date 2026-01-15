//! Core types layer - foundational types used across all layers

mod page_types;
mod delta_types;
mod epoch_types;
mod gc;

pub use page_types::{PageID, PageLocation, PageError};
pub use delta_types::{DeltaID, Source, DeltaError};
pub use epoch_types::{Epoch, EpochCell};
pub use gc::{GCMetrics, MemoryPressureHandler};
