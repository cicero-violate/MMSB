//! Proofs Module - Re-exports canonical proofs from mmsb-proof
//!
//! Only A–G proofs from the canonical spec are allowed here.
//! No legacy Mmsb* types, no file I/O, no streams.

pub use mmsb_proof::{
    AdmissionProof,
    CommitProof,
    Hash,
    IntentProof,
    JudgmentProof,
    OutcomeProof,
    PolicyProof,
    KnowledgeProof,
    Proof,
    ProduceProof,
    IntentStage,
    PolicyStage,
    JudgmentStage,
    AdmissionStage,
    CommitStage,
    OutcomeStage,
    KnowledgeStage,
};

// NO legacy re-exports — remove all MmsbAdmissionProof, MmsbExecutionProof, etc.
// If you need custom admission logic, implement it using canonical AdmissionProof
