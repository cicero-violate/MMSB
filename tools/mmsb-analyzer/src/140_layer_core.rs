//! Core layer computation functions - lowest level utilities
//! This module must have no dependencies on other local modules to avoid circular dependencies.
//! It provides fundamental layer prefix extraction and violation detection.

#[allow(unused_imports)]
pub use crate::cluster_006::{
    layer_prefix_value, order_directories, collect_directory_moves,
};
#[allow(unused_imports)]
pub use crate::cluster_008::detect_layer_violations;
#[allow(unused_imports)]
pub use crate::cluster_001::{layer_constrained_sort, topo_sort_within};
pub use crate::cluster_008::sort_structural_items;





// Moved to layer_utilities in Batch 4 - no re-export needed
