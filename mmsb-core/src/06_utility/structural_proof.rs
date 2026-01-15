use sha2::{Digest, Sha256};
use crate::dag::StructuralOp;

pub const STRUCTURAL_PROOF_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct MmsbStructuralAdmissionProof {
    pub version: u32,
    pub ops_hash: String,
    pub dag_snapshot_hash: Option<String>,
    pub conversation_id: String,
    pub message_id: String,
    pub scope: String,
    pub approved: bool,
    pub epoch: u64,
}

impl MmsbStructuralAdmissionProof {
    pub fn new(
        ops: &[StructuralOp],
        dag_snapshot_hash: Option<String>,
        conversation_id: String,
        message_id: String,
        scope: String,
        approved: bool,
        epoch: u64,
    ) -> Self {
        let ops_hash = compute_ops_hash(ops);
        Self {
            version: STRUCTURAL_PROOF_VERSION,
            ops_hash,
            dag_snapshot_hash,
            conversation_id,
            message_id,
            scope,
            approved,
            epoch,
        }
    }

    pub fn verify_ops(&self, ops: &[StructuralOp]) -> bool {
        let computed = compute_ops_hash(ops);
        computed == self.ops_hash
    }
}

fn compute_ops_hash(ops: &[StructuralOp]) -> String {
    let mut hasher = Sha256::new();
    for op in ops {
        let s = format!("{:?}", op);
        hasher.update(s.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}
