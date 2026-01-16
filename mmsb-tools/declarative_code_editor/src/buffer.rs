use crate::error::EditorError;
use crate::types::Edit;
use syn::File;

/// EditBuffer maintains source code and pending edits
#[derive(Debug, Clone)]
pub struct EditBuffer {
    source: String,
    tree: File,
    edits: Vec<Edit>,
    last_diff: Option<String>,
}

impl EditBuffer {
    /// Create a new buffer from source code
    pub fn new(source: impl Into<String>) -> Result<Self, EditorError> {
        let source = source.into();
        let tree = syn::parse_file(&source)
            .map_err(|e| EditorError::ParseError(e.to_string()))?;
        
        Ok(Self {
            source,
            tree,
            edits: Vec::new(),
            last_diff: None,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn tree(&self) -> &File {
        &self.tree
    }

    pub fn edits(&self) -> &[Edit] {
        &self.edits
    }

    pub fn has_edits(&self) -> bool {
        !self.edits.is_empty()
    }

    pub fn add_edit(&mut self, edit: Edit) {
        self.edits.push(edit);
    }

    pub fn clear_edits(&mut self) {
        self.edits.clear();
        self.last_diff = None;
    }

    pub fn last_diff(&self) -> Option<&str> {
        self.last_diff.as_deref()
    }

    /// Apply pending edits and reparse
    pub fn apply_edits(&mut self) -> Result<String, EditorError> {
        if self.edits.is_empty() {
            return Ok(self.source.clone());
        }

        // Sort edits by start_byte descending (apply from end to start)
        let mut edits = self.edits.clone();
        edits.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

        let mut result = self.source.clone();
        for edit in edits {
            let start = edit.start_byte as usize;
            let end = edit.old_end_byte as usize;
            result.replace_range(start..end, &edit.new_text);
        }

        // Reparse
        self.tree = syn::parse_file(&result)
            .map_err(|e| EditorError::ParseError(e.to_string()))?;
        self.source = result.clone();
        self.edits.clear();

        Ok(result)
    }

    /// Render current state (with pending edits applied)
    pub fn render(&self) -> String {
        if self.edits.is_empty() {
            return self.source.clone();
        }

        let mut edits = self.edits.clone();
        edits.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

        let mut result = self.source.clone();
        for edit in edits {
            let start = edit.start_byte as usize;
            let end = edit.old_end_byte as usize;
            result.replace_range(start..end, &edit.new_text);
        }
        result
    }
}
