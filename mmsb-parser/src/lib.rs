//! MMSB Parser - LLM Response → Structured Intents
//!
//! Extracts code blocks, commands, and intents from LLM responses.
//! Converts unstructured text into structured Intent objects for JudgmentBus.
//!
//! Authority: NONE (pure parsing)

pub mod codeblock;
pub mod intent_extractor;
pub mod mmsb_parser;

pub use codeblock::{CodeBlock, CodeBlockExtractor};
pub use intent_extractor::IntentExtractor;
pub use mmsb_parser::MMSBParser;
