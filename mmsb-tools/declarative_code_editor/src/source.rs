//! In-memory source representation
//!
//! Holds Rust source code in memory for querying and mutation.
//! Similar to structural_code_indexer's SourceFile but with AST caching.

use syn::File;
use std::path::PathBuf;
use crate::error::EditorError;

/// In-memory source file
#[derive(Debug, Clone)]
pub struct SourceBuffer {
    /// File path (for identification)
    pub path: PathBuf,
    
    /// Source content (in-memory)
    pub content: String,
    
    /// Parsed AST (cached)
    ast: File,
}

impl SourceBuffer {
    /// Create from path and content
    pub fn new(path: PathBuf, content: String) -> Result<Self, EditorError> {
        let ast = syn::parse_file(&content)
            .map_err(|e| EditorError::ParseError(e.to_string()))?;
        
        Ok(Self { path, content, ast })
    }
    
    /// Get AST
    pub fn ast(&self) -> &File {
        &self.ast
    }
    
    /// Get source content
    pub fn source(&self) -> &str {
        &self.content
    }
    
    /// Update content (re-parses AST)
    pub fn update(&mut self, new_content: String) -> Result<(), EditorError> {
        let new_ast = syn::parse_file(&new_content)
            .map_err(|e| EditorError::ParseError(e.to_string()))?;
        
        self.content = new_content;
        self.ast = new_ast;
        Ok(())
    }
}
