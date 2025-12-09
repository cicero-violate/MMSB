//! Layer 5: Adaptive Memory
//! 
//! Self-optimizing memory layout and page clustering

pub mod memory_layout;
pub mod page_clustering;
pub mod locality_optimizer;

pub use memory_layout::{MemoryLayout, AccessPattern, PageId, PhysAddr};
pub use page_clustering::{PageCluster, PageClusterer};
pub use locality_optimizer::LocalityOptimizer;
