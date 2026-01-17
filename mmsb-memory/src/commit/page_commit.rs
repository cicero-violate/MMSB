//! Page Commit - Commit delta to page
use crate::tlog::{self, TransactionLog};
use crate::delta::Delta;
use crate::proofs::{ADMISSION_PROOF_VERSION, MmsbAdmissionProof};
use crate::dag::DependencyGraph;
pub(crate) fn commit_delta(
log: &TransactionLog,
admission_proof: &MmsbAdmissionProof,
delta: Delta,
active_dag: Option<&DependencyGraph>,
) -> std::io::Result<()> {
if admission_proof.version != ADMISSION_PROOF_VERSION {
return Err(std::io::Error::new(
std::io::ErrorKind::PermissionDenied,
"admission proof version mismatch",
));
}
let expected_hash = tlog::delta_hash(&delta);
if admission_proof.delta_hash != expected_hash {
return Err(std::io::Error::new(
std::io::ErrorKind::PermissionDenied,
"admission proof hash mismatch",
));
}
if let Some(dag) = active_dag {
if let Some(proof_hash) = &admission_proof.dag_snapshot_hash {
let dag_hash = dag.compute_snapshot_hash();
if proof_hash != &dag_hash {
return Err(std::io::Error::new(
std::io::ErrorKind::PermissionDenied,
format!("DAG snapshot hash mismatch: proof={}, actual={}", proof_hash, dag_hash),
));
}
} else {
eprintln!("WARNING: Legacy admission proof without DAG snapshot hash");
}
}
log.append(admission_proof, delta)
}
#[cfg(test)]
mod tests {
use super::commit_delta;
use crate::delta::Delta;
use crate::tlog::{self, TransactionLog};
use crate::proofs::{ADMISSION_PROOF_VERSION, MmsbAdmissionProof};
use std::time::{SystemTime, UNIX_EPOCH};
fn base_delta() -> Delta {
// ... your existing base_delta ...
}
fn admission_proof(delta_hash: String, version: u32) -> MmsbAdmissionProof {
MmsbAdmissionProof {
version,
delta_hash,
dag_snapshot_hash: None,
conversation_id: "test".to_string(),
message_id: "test".to_string(),
suffix: "0".to_string(),
intent_hash: "test".to_string(),
approved: true,
command: Vec::new(),
cwd: None,
env: None,
epoch: 0,
}
}
#[test]
fn commit_without_valid_admission_proof_halts() {
let nanos = SystemTime::now()
.duration_since(UNIX_EPOCH)
.unwrap()
.as_nanos();
let path = std::env::temp_dir().join(format!("mmsb_commit_no_proof_{nanos}.tlog"));
let log = TransactionLog::new(&path).expect("log");
let delta = base_delta();
let delta_hash = tlog::delta_hash(&delta);
let admission = admission_proof("bad-hash".to_string(), ADMISSION_PROOF_VERSION);
let err = commit_delta(&log, &admission, delta, None)
.expect_err("expected admission proof failure");
assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
}
#[test]
fn commit_with_valid_admission_proof_succeeds() {
let nanos = SystemTime::now()
.duration_since(UNIX_EPOCH)
.unwrap()
.as_nanos();
let path = std::env::temp_dir().join(format!("mmsb_commit_with_proof_{nanos}.tlog"));
let log = TransactionLog::new(&path).expect("log");
let delta = base_delta();
let delta_hash = tlog::delta_hash(&delta);
let admission = admission_proof(delta_hash, ADMISSION_PROOF_VERSION);
commit_delta(&log, &admission, delta, None)
.expect("commit succeeds");
}
}
