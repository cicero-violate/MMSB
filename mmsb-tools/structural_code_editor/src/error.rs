use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodeEditError {
    #[error("duplicate file path in snapshot: {0}")]
    DuplicatePath(String),
    #[error("active DAG does not match scanned snapshot (expected {expected}, found {found})")]
    DagMismatch { expected: String, found: String },
}
