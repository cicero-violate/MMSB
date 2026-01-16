//! Bridge Layer
//!
//! Connects declarative_code_editor to structural_code_editor
//! and the MMSB authority model.
//!
//! Flow:
//! 1. DeclarativeCodeEditor produces buffer edits
//! 2. IntentBridge extracts semantic intent
//! 3. StructuralClassifier routes to DAG/State pipelines
//! 4. PropagationBridge triggers propagation engine

pub mod intent_bridge;
pub mod structural_classifier;
pub mod propagation_bridge;
pub mod output;
pub mod orchestrator;

pub use intent_bridge::IntentBridge;
pub use structural_classifier::StructuralClassifier;
pub use propagation_bridge::PropagationBridge;
pub use output::{BridgedOutput, PipelineRoute};
pub use orchestrator::{BridgeOrchestrator, BridgedOutputWithPropagation};
