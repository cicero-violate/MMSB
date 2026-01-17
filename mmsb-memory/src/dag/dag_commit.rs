use crate::dag::{
    DagCommitError, DependencyGraph, StructuralOp,
    has_cycle, GraphValidator,
};
use super::shadow_graph::ShadowPageGraph;
use crate::dag::dag_log::{append_structural_record, default_structural_log_path};
use crate::proofs::{MmsbStructuralAdmissionProof, STRUCTURAL_PROOF_VERSION};
use mmsb_judgment::JudgmentToken;

pub fn build_dependency_graph(ops: &[StructuralOp]) -> DependencyGraph {
    let mut dag = DependencyGraph::new();
    dag.apply_ops(ops);
    dag
}

pub(crate) fn apply_ops_unchecked(dag: &mut DependencyGraph, ops: &[StructuralOp]) {
    dag.apply_ops(ops);
}

pub fn commit_structural_delta(
    dag: &mut DependencyGraph,
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

    let shadow = ShadowPageGraph::from_dag_and_ops(dag, ops);

    if has_cycle(&shadow) {
        return Err(DagCommitError::CycleDetected);
    }

    let validator = GraphValidator::new(&shadow);
    let report = validator.detect_cycles();
    if report.has_cycle {
        return Err(DagCommitError::CycleDetected);
    }

    let log_path = default_structural_log_path()?;
    append_structural_record(&log_path, ops, proof)?;

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
    use crate::page::PageID;
    use crate::dag::EdgeType;

    #[test]
    fn commit_rejects_empty_ops() {
        let mut dag = DependencyGraph::new();
        let ops = vec![];
        let proof = MmsbStructuralAdmissionProof::new(
            &ops,
            None,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            true,
            0,
        );
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &ops, &token, &proof)
            .expect_err("expected empty ops error");
        assert!(matches!(err, DagCommitError::EmptyOperations));
    }

    #[test]
    fn commit_rejects_invalid_proof_version() {
        let mut dag = DependencyGraph::new();
        let ops = vec![StructuralOp::AddEdge {
            from: PageID(1),
            to: PageID(2),
            edge_type: EdgeType::Data,
        }];
        let mut proof = MmsbStructuralAdmissionProof::new(
            &ops,
            None,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            true,
            0,
        );
        proof.version = 999;
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &ops, &token, &proof)
            .expect_err("expected version mismatch");
        assert!(matches!(err, DagCommitError::ProofVersionMismatch { .. }));
    }

    #[test]
    fn commit_rejects_unapproved_proof() {
        let mut dag = DependencyGraph::new();
        let ops = vec![StructuralOp::AddEdge {
            from: PageID(1),
            to: PageID(2),
            edge_type: EdgeType::Data,
        }];
        let proof = MmsbStructuralAdmissionProof::new(
            &ops,
            None,
            "test_conv".to_string(),
            "test_msg".to_string(),
            "test_scope".to_string(),
            false,
            0,
        );
        let token = JudgmentToken::test_only();

        let err = commit_structural_delta(&mut dag, &ops, &token, &proof)
            .expect_err("expected proof not approved");
        assert!(matches!(err, DagCommitError::ProofNotApproved));
    }
}
