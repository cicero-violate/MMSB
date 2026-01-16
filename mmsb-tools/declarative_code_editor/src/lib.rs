//! Declarative Code Editor
//!
//! A declarative DSL for querying and mutating Rust AST with MMSB authority model.
//!
//! ## Architecture
//!
//! ```text
//! Query DSL → Mutation DSL → Upsert Engine → Pure Observation
//!                                              ↓
//!                                         Intent Extraction
//!                                              ↓
//!                                         Judgment Gate
//!                                              ↓
//!                                         Propagation
//!                                              ↓
//!                                    Dual Commit (State + Structural)
//! ```
//!
//! ## Pipelines
//!
//! **STRUCTURAL**: Changes DAG causality (what depends on what)
//! **STATE**: Changes page content (what the code says)

pub mod buffer;
pub mod query;
pub mod mutation;
pub mod upsert;
pub mod intent;
pub mod executor;
pub mod error;
pub mod types;
pub mod bridge;

pub use buffer::EditBuffer;
pub use query::{Predicate, KindPredicate, NamePredicate, CustomPredicate, QueryPlan};
pub use mutation::{MutationOp, ReplaceOp, WrapOp, DeleteOp, InsertOp, MutationPlan};
pub use upsert::{UpsertSpec, UpsertResult, AnchorSpec};
pub use intent::{EditIntent, IntentCategory};
pub use executor::{execute_query, plan_mutations, execute_upsert};
pub use error::EditorError;
pub use types::{EditorOutput, Edit};
pub use bridge::{
    BridgeOrchestrator, BridgedOutput, BridgedOutputWithPropagation, PipelineRoute,
};
