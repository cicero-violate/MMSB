//! Intent extraction from code blocks → structured Intent objects

use crate::CodeBlockExtractor;
use mmsb_events::Intent;

pub struct IntentExtractor {
    extractor: CodeBlockExtractor,
}

impl IntentExtractor {
    pub fn new() -> Self {
        Self {
            extractor: CodeBlockExtractor::new(),
        }
    }
    
    /// Extract intents from LLM response text
    pub fn extract_intents(&self, text: &str) -> Vec<Intent> {
        let blocks = self.extractor.extract(text);
        let mut intents = Vec::new();
        
        for block in blocks {
            let intent = match block.language.as_str() {
                "bash" | "sh" | "shell" => {
                    self.create_shell_intent(&block.content)
                }
                "rust" | "python" | "javascript" | "typescript" => {
                    self.create_code_intent(&block.language, &block.content)
                }
                _ => continue,
            };
            
            intents.push(intent);
        }
        
        intents
    }
    
    fn create_shell_intent(&self, command: &str) -> Intent {
        // Analyze shell command to extract metadata
        let lines: Vec<&str> = command.lines().collect();
        let tools_used = self.extract_tools_from_shell(command);
        
        Intent {
            description: format!("Execute shell command: {}", lines[0]),
            intent_class: "shell".to_string(),
            target_paths: vec![],
            tools_used,
            files_touched: 0,
            diff_lines: lines.len(),
            max_duration_ms: 60000, // 60 seconds default
            max_memory_bytes: 512 * 1024 * 1024, // 512MB default
        }
    }
    
    fn create_code_intent(&self, language: &str, code: &str) -> Intent {
        let lines: Vec<&str> = code.lines().collect();
        
        Intent {
            description: format!("Apply {} code change", language),
            intent_class: format!("{}_edit", language),
            target_paths: vec![],
            tools_used: vec![language.to_string()],
            files_touched: 1,
            diff_lines: lines.len(),
            max_duration_ms: 5000,
            max_memory_bytes: 256 * 1024 * 1024,
        }
    }
    
    fn extract_tools_from_shell(&self, command: &str) -> Vec<String> {
        // Extract command names from shell script
        let mut tools = Vec::new();
        
        for line in command.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            
            // Extract first word as tool name
            if let Some(tool) = trimmed.split_whitespace().next() {
                if !tool.starts_with("$") && !tool.starts_with("./") {
                    tools.push(tool.to_string());
                }
            }
        }
        
        tools.dedup();
        tools
    }
}

impl Default for IntentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_shell_intent() {
        let extractor = IntentExtractor::new();
        let text = r#"
Run this command:

```bash
cargo build --release
```
"#;
        
        let intents = extractor.extract_intents(text);
        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].intent_class, "shell");
        assert!(intents[0].tools_used.contains(&"cargo".to_string()));
    }
}
