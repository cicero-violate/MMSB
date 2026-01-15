use crate::dag::{
    DagCommitError, DependencyGraph, ShadowPageGraph, StructuralOp,
    has_cycle, GraphValidator,
};
use crate::utility::{MmsbStructuralAdmissionProof, STRUCTURAL_PROOF_VERSION};
use mmsb_judgment::JudgmentToken;

pub fn commit_structural_delta(
    dag: &mut DependencyGraph,
    shadow: &ShadowPageGraph,
    ops: &[StructuralOp],
    token: &JudgmentToken,
    proof: &MmsbStructuralAdmissionProof,
) -> Result<(), DagCommitError> {
    let _ = token;

    if ops.is_empty() {
        return Err(DagCommitError::EmptyOperations);
    }

    if proof.version != STRUCTURAL_PROOF_VERSION {
        return Err(DagCommitError::ProofVersionMismatch {
            expected: STRUCTURAL_PROOF_VERSION,
            found: proof.version,
        });
    }

    if !proof.verify_ops(ops) {
        let expected = proof.ops_hash.clone();
        let found = compute_ops_hash_local(ops);
        return Err(DagCommitError::ProofHashMismatch { expected, found });
    }

    if !proof.approved {
        return Err(DagCommitError::ProofNotApproved);
    }

    for op in ops {
        if !dag.contains_page(op.from_page()) {
            return Err(DagCommitError::InvalidPageReference(op.from_page()));
        }
        if !dag.contains_page(op.to_page()) {
            return Err(DagCommitError::InvalidPageReference(op.to_page()));
        }
    }

    if has_cycle(shadow) {
        return Err(DagCommitError::CycleDetected);
    }

    let validator = GraphValidator::new(shadow);
    let report = validator.detect_cycles();
    if report.has_cycle {
        return Err(DagCommitError::CycleDetected);
    }

    dag.apply_ops(ops);

    Ok(())
}

fn compute_ops_hash_local(ops: &[StructuralOp]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for op in ops {
        let s = format!("{:?}", op);
        hasher.update(s.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PageID;
    use crate::dag::EdgeType;

    #[test]
    fn commit_rejects_empty_ops() {
        let mut dag = DependencyGraph::new();
        let shadow = ShadowPageGraph::default();
        let ops = vec![];
        let proof = MmsbStructuralAdmissionProof::new(
            &ops,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            true,
            0,
        );
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &shadow, &ops, &token, &proof)
            .expect_err("expected empty ops error");
        assert!(matches!(err, DagCommitError::EmptyOperations));
    }

    #[test]
    fn commit_rejects_invalid_proof_version() {
        let mut dag = DependencyGraph::new();
        let shadow = ShadowPageGraph::default();
        let ops = vec![StructuralOp::AddEdge {
            from: PageID(1),
            to: PageID(2),
            edge_type: EdgeType::Data,
        }];
        let mut proof = MmsbStructuralAdmissionProof::new(
            &ops,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            true,
            0,
        );
        proof.version = 999;
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &shadow, &ops, &token, &proof)
            .expect_err("expected version mismatch");
        assert!(matches!(err, DagCommitError::ProofVersionMismatch { .. }));
    }

    #[test]
    fn commit_rejects_unapproved_proof() {
        let mut dag = DependencyGraph::new();
        let shadow = ShadowPageGraph::default();
        let ops = vec![StructuralOp::AddEdge {
            from: PageID(1),
            to: PageID(2),
            edge_type: EdgeType::Data,
        }];
        let proof = MmsbStructuralAdmissionProof::new(
            &ops,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            false,
            0,
        );
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &shadow, &ops, &token, &proof)
            .expect_err("expected proof not approved");
        assert!(matches!(err, DagCommitError::ProofNotApproved));
    }
}
