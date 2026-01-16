use mmsb_core::dag::StructuralOp;
use mmsb_core::prelude::Delta;

/// Edit represents a byte-level change to source code
#[derive(Debug, Clone)]
pub struct Edit {
    pub start_byte: u32,
    pub old_end_byte: u32,
    pub new_text: String,
}

impl Edit {
    pub fn new(start_byte: u32, old_end_byte: u32, new_text: impl Into<String>) -> Self {
        Self {
            start_byte,
            old_end_byte,
            new_text: new_text.into(),
        }
    }

    pub fn byte_range(&self) -> (u32, u32) {
        (self.start_byte, self.old_end_byte)
    }
}

/// Output from declarative editor - separated by pipeline
#[derive(Debug, Clone, Default)]
pub struct EditorOutput {
    /// STATE PIPELINE: Page content changes
    pub page_deltas: Vec<Delta>,
    
    /// STRUCTURAL PIPELINE: DAG causality changes
    pub structural_ops: Vec<StructuralOp>,
}

impl EditorOutput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_deltas(mut self, deltas: Vec<Delta>) -> Self {
        self.page_deltas = deltas;
        self
    }

    pub fn with_ops(mut self, ops: Vec<StructuralOp>) -> Self {
        self.structural_ops = ops;
        self
    }
}
