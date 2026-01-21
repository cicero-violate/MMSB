//! Parser Bus - Content extraction and classification
//!
//! Parses LLM responses into structured intents:
//! - Shell commands
//! - Patch files  
//! - JSON commands
//! - Code blocks (Rust, Python, Julia, etc.)
//! - Plan documents

use mmsb_primitives::{EventId, Timestamp, Hash};
use mmsb_proof::JudgmentProof;
use serde::{Deserialize, Serialize};

/// Content block type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    Patch,
    Shell,
    Plan,
    Rust,
    Python,
    Julia,
    Json,
    Yaml,
    Toml,
    Markdown,
    Text,
    Other,
}

/// Parsed code block from LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: Option<String>,
    pub content: String,
    pub block_type: BlockType,
}

/// Parsed content from LLM message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedContent {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub source_message_id: String,
    pub source_conversation_id: String,
    pub blocks: Vec<CodeBlock>,
}

/// Shell command intent extracted from parsed content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellIntent {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub command: String,
    pub working_directory: Option<String>,
    pub source_block_index: usize,
}

/// Patch intent extracted from parsed content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchIntent {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub patch_content: String,
    pub target_file: Option<String>,
    pub source_block_index: usize,
}

/// JSON command intent extracted from parsed content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonIntent {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub json_content: String,
    pub command_type: Option<String>,
    pub source_block_index: usize,
}

/// Plan document extracted from parsed content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanIntent {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub plan_content: String,
    pub source_block_index: usize,
}

/// Parser trait - extracts structured intents from text
pub trait Parser: Send + Sync {
    /// Parse raw text into structured code blocks
    fn parse(&self, text: &str) -> Vec<CodeBlock>;
    
    /// Classify a code block's type
    fn classify_block(&self, block: &CodeBlock) -> BlockType;
    
    /// Extract shell intents from parsed content
    fn extract_shell_intents(&self, parsed: &ParsedContent) -> Vec<ShellIntent>;
    
    /// Extract patch intents from parsed content
    fn extract_patch_intents(&self, parsed: &ParsedContent) -> Vec<PatchIntent>;
    
    /// Extract JSON intents from parsed content
    fn extract_json_intents(&self, parsed: &ParsedContent) -> Vec<JsonIntent>;
    
    /// Extract plan intents from parsed content
    fn extract_plan_intents(&self, parsed: &ParsedContent) -> Vec<PlanIntent>;
}

/// ParserBus - event-driven parser interface
pub trait ParserBus: Send + Sync {
    /// Submit raw text for parsing
    fn submit_for_parsing(
        &mut self,
        conversation_id: String,
        message_id: String,
        text: String,
    ) -> EventId;
    
    /// Emit parsed content event
    fn emit_parsed(&mut self, parsed: ParsedContent);
    
    /// Emit shell intent event
    fn emit_shell_intent(&mut self, intent: ShellIntent);
    
    /// Emit patch intent event  
    fn emit_patch_intent(&mut self, intent: PatchIntent);
    
    /// Emit JSON intent event
    fn emit_json_intent(&mut self, intent: JsonIntent);
    
    /// Emit plan intent event
    fn emit_plan_intent(&mut self, intent: PlanIntent);
}

/// Parse request - submitted to parser
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseRequest {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub conversation_id: String,
    pub message_id: String,
    pub text: String,
}

/// Parse result - emitted after parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub request_id: EventId,
    pub parsed_content: ParsedContent,
    pub shell_intents: Vec<ShellIntent>,
    pub patch_intents: Vec<PatchIntent>,
    pub json_intents: Vec<JsonIntent>,
    pub plan_intents: Vec<PlanIntent>,
}
