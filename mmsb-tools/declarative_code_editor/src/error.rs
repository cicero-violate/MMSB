use thiserror::Error;

#[derive(Debug, Error)]
pub enum EditorError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Query returned no matches")]
    NoMatches,

    #[error("Query returned {0} matches; expected single result")]
    MultipleMatches(usize),

    #[error("Conflicting edits at byte ranges {0}:{1} and {2}:{3}")]
    ConflictingEdits(u32, u32, u32, u32),

    #[error("Buffer has pending edits; clean buffer required")]
    DirtyBuffer,

    #[error("Invalid anchor: {0}")]
    InvalidAnchor(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Intent extraction failed: {0}")]
    IntentExtractionError(String),
}
