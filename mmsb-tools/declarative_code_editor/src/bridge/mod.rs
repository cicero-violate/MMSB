//! Bridge Layer
//!
//! Connects declarative_code_editor to structural_code_editor
//! and the MMSB authority model.
//!
//! Flow:
//! 1. DeclarativeCodeEditor produces buffer edits
//! 2. IntentBridge extracts semantic intent
//! 3. StructuralClassifier routes to DAG/State pipelines

pub mod structural_classifier;
pub mod output;
pub mod orchestrator;

pub use structural_classifier::StructuralClassifier;
pub use output::{BridgedOutput, PipelineRoute};
pub use orchestrator::BridgeOrchestrator;
