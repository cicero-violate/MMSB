//! MMSB Parser Implementation - implements Parser trait from mmsb-events

use mmsb_events::{
    Parser, ParserBus, CodeBlock as EventCodeBlock, BlockType, ParsedContent,
    ShellIntent, PatchIntent, JsonIntent, PlanIntent, ParseRequest, ParseResult,
};
use mmsb_primitives::{EventId, Timestamp};
use crate::codeblock::{CodeBlock, CodeBlockExtractor};
use std::time::{SystemTime, UNIX_EPOCH};

/// MMSB Parser implementation
pub struct MMSBParser {
    extractor: CodeBlockExtractor,
}

impl MMSBParser {
    pub fn new() -> Self {
        Self {
            extractor: CodeBlockExtractor::new(),
        }
    }
}

impl Default for MMSBParser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser for MMSBParser {
    fn parse(&self, text: &str) -> Vec<EventCodeBlock> {
        let blocks = self.extractor.extract(text);
        blocks
            .into_iter()
            .map(|block| {
                let block_type = self.classify_by_language(&Some(block.language.clone()));
                EventCodeBlock {
                    language: Some(block.language),
                    content: block.content,
                    block_type,
                }
            })
            .collect()
    }

    fn classify_block(&self, block: &EventCodeBlock) -> BlockType {
        self.classify_by_language(&block.language)
    }

    fn extract_shell_intents(&self, parsed: &ParsedContent) -> Vec<ShellIntent> {
        let mut intents = Vec::new();
        for (idx, block) in parsed.blocks.iter().enumerate() {
            if block.block_type == BlockType::Shell {
                intents.push(ShellIntent {
                    event_id: parsed.event_id,
                    timestamp: current_timestamp(),
                    command: block.content.clone(),
                    working_directory: None,
                    source_block_index: idx,
                });
            }
        }
        intents
    }

    fn extract_patch_intents(&self, parsed: &ParsedContent) -> Vec<PatchIntent> {
        let mut intents = Vec::new();
        for (idx, block) in parsed.blocks.iter().enumerate() {
            if block.block_type == BlockType::Patch {
                intents.push(PatchIntent {
                    event_id: parsed.event_id,
                    timestamp: current_timestamp(),
                    patch_content: block.content.clone(),
                    target_file: None,
                    source_block_index: idx,
                });
            }
        }
        intents
    }

    fn extract_json_intents(&self, parsed: &ParsedContent) -> Vec<JsonIntent> {
        let mut intents = Vec::new();
        for (idx, block) in parsed.blocks.iter().enumerate() {
            if block.block_type == BlockType::Json {
                intents.push(JsonIntent {
                    event_id: parsed.event_id,
                    timestamp: current_timestamp(),
                    json_content: block.content.clone(),
                    command_type: None,
                    source_block_index: idx,
                });
            }
        }
        intents
    }

    fn extract_plan_intents(&self, parsed: &ParsedContent) -> Vec<PlanIntent> {
        let mut intents = Vec::new();
        for (idx, block) in parsed.blocks.iter().enumerate() {
            if block.block_type == BlockType::Plan {
                intents.push(PlanIntent {
                    event_id: parsed.event_id,
                    timestamp: current_timestamp(),
                    plan_content: block.content.clone(),
                    source_block_index: idx,
                });
            }
        }
        intents
    }
}

impl MMSBParser {
    fn classify_by_language(&self, language: &Option<String>) -> BlockType {
        match language.as_ref().map(|s| s.to_lowercase()) {
            Some(ref lang) if lang == "diff" || lang == "patch" => BlockType::Patch,
            Some(ref lang) if lang == "sh" || lang == "bash" || lang == "shell" => BlockType::Shell,
            Some(ref lang) if lang == "plan" => BlockType::Plan,
            Some(ref lang) if lang == "rust" || lang == "rs" => BlockType::Rust,
            Some(ref lang) if lang == "python" || lang == "py" => BlockType::Python,
            Some(ref lang) if lang == "julia" || lang == "jl" => BlockType::Julia,
            Some(ref lang) if lang == "json" => BlockType::Json,
            Some(ref lang) if lang == "yaml" || lang == "yml" => BlockType::Yaml,
            Some(ref lang) if lang == "toml" => BlockType::Toml,
            Some(ref lang) if lang == "markdown" || lang == "md" => BlockType::Markdown,
            Some(ref lang) if lang == "text" || lang == "txt" => BlockType::Text,
            None => BlockType::Text,
            _ => BlockType::Other,
        }
    }
}

fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
