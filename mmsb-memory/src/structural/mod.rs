pub mod structural_types;
pub mod shadow_graph;
pub mod shadow_graph_mod;
pub mod shadow_graph_traversal;
pub mod cycle_detection;
pub mod graph_validator;

pub use structural_types::StructuralOp;
pub use cycle_detection::has_cycle;
pub(crate) use graph_validator::GraphValidator;
pub use graph_validator::GraphValidationReport;
