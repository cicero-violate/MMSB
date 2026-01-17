//! Optimization layer (legacy/locality helpers).

pub mod memory_layout;
pub mod page_clustering;
pub mod locality_optimizer;

pub use memory_layout::{AccessPattern, MemoryLayout, PageId, PhysAddr};
pub use page_clustering::{PageCluster, PageClusterer};
pub use locality_optimizer::LocalityOptimizer;
