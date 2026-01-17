use crate::types::PageID;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagCommitError {
    CycleDetected,
    InvalidPageReference(PageID),
    ProofVersionMismatch { expected: u32, found: u32 },
    ProofHashMismatch { expected: String, found: String },
    ProofNotApproved,
    InvalidJudgment,
    ValidationFailed(String),
    PersistenceError(String),
    EmptyOperations,
}
