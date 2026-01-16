//! Declarative Code Editor
//!
//! Declarative DSL for querying and mutating Rust AST.
//! Produces Deltas and StructuralOps for MMSB authority model.
//!
//! ## Architecture
//!
//! ```text
//! Source Text → Query (AST match) → Mutation (transform) → Delta
//!                                                            ↓
//!                                                    Intent Extraction
//!                                                            ↓
//!                                                    Classify: State vs Structural
//!                                                            ↓
//!                                                    BridgedOutput
//! ```

pub mod source;
pub mod query;
pub mod mutation;
pub mod intent;
pub mod executor;
pub mod error;
pub mod bridge;
pub mod propagation;

pub use source::SourceBuffer;
pub use query::{Predicate, KindPredicate, NamePredicate, CustomPredicate, QueryPlan};
pub use query::ItemKind;
pub use mutation::{MutationOp, ReplaceOp, WrapOp, DeleteOp, InsertOp, MutationPlan};
pub use intent::{EditIntent, IntentCategory};
pub use executor::{execute_query, apply_mutation};
pub use error::EditorError;
pub use bridge::{
  BridgeOrchestrator, BridgedOutput, PipelineRoute,
};
pub use propagation::{
    propagate_from_buffer, source_buffer_to_page_index, translate_intent,
    propagator::PropagatedDelta,
};
