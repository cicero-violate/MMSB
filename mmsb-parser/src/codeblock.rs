//! Code block extraction from markdown-formatted LLM responses

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub language: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

pub struct CodeBlockExtractor {
    fence_regex: Regex,
}

impl CodeBlockExtractor {
    pub fn new() -> Self {
        Self {
            fence_regex: Regex::new(r"^```(\w+)?$").unwrap(),
        }
    }
    
    /// Extract all code blocks from markdown text
    pub fn extract(&self, text: &str) -> Vec<CodeBlock> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        
        let mut in_block = false;
        let mut current_language = String::new();
        let mut current_content = String::new();
        let mut start_line = 0;
        
        for (i, line) in lines.iter().enumerate() {
            if let Some(caps) = self.fence_regex.captures(line) {
                if !in_block {
                    // Start of code block
                    in_block = true;
                    current_language = caps.get(1)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_else(|| "plaintext".to_string());
                    current_content.clear();
                    start_line = i + 1;
                } else {
                    // End of code block
                    blocks.push(CodeBlock {
                        language: current_language.clone(),
                        content: current_content.trim_end().to_string(),
                        start_line,
                        end_line: i,
                    });
                    in_block = false;
                }
            } else if in_block {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }
        }
        
        blocks
    }
    
    /// Extract shell commands specifically
    pub fn extract_shell_commands(&self, text: &str) -> Vec<String> {
        self.extract(text)
            .into_iter()
            .filter(|block| {
                matches!(
                    block.language.as_str(),
                    "bash" | "sh" | "shell" | "zsh" | "fish"
                )
            })
            .map(|block| block.content)
            .collect()
    }
    
    /// Extract code blocks by language
    pub fn extract_by_language(&self, text: &str, language: &str) -> Vec<CodeBlock> {
        self.extract(text)
            .into_iter()
            .filter(|block| block.language.eq_ignore_ascii_case(language))
            .collect()
    }
}

impl Default for CodeBlockExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_single_block() {
        let extractor = CodeBlockExtractor::new();
        let text = r#"
Here's a Python example:

```python
def hello():
    print("Hello, world!")
```

That's it!
"#;
        
        let blocks = extractor.extract(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].language, "python");
        assert!(blocks[0].content.contains("def hello()"));
    }
    
    #[test]
    fn test_extract_multiple_blocks() {
        let extractor = CodeBlockExtractor::new();
        let text = r#"
```rust
fn main() {
    println!("Rust");
}
```

And in Python:

```python
print("Python")
```
"#;
        
        let blocks = extractor.extract(text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].language, "rust");
        assert_eq!(blocks[1].language, "python");
    }
}
