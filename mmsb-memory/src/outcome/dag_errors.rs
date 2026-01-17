use crate::page::PageID;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DagCommitError {
    #[error("cycle detected in proposed DAG structure")]
    CycleDetected,

    #[error("invalid edge reference: page {0} does not exist")]
    InvalidPageReference(PageID),

    #[error("structural proof version mismatch: expected {expected}, found {found}")]
    ProofVersionMismatch { expected: u32, found: u32 },

    #[error("structural proof hash mismatch: expected {expected}, found {found}")]
    ProofHashMismatch { expected: String, found: String },

    #[error("structural proof not approved")]
    ProofNotApproved,

    #[error("judgment token invalid or missing")]
    InvalidJudgment,

    #[error("DAG validation failed: {0}")]
    ValidationFailed(String),

    #[error("persistence error: {0}")]
    PersistenceError(#[from] std::io::Error),

    #[error("empty operation set")]
    EmptyOperations,
}
