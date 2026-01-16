//! Propagation Integration
//!
//! Connects declarative_code_editor to structural_code_editor's propagation engine.
//!
//! Flow:
//! 1. Convert SourceBuffer → PageIndex
//! 2. Translate EditIntent (declarative → structural)
//! 3. Call structural_code_editor::propagate_edits
//! 4. Return derived deltas for affected pages

pub mod conversion;
pub mod intent_translator;
pub mod propagator;

pub use conversion::source_buffer_to_page_index;
pub use intent_translator::translate_intent;
pub use propagator::propagate_from_buffer;
