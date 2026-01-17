use crate::tlog::{self, TransactionLog};
use crate::delta::Delta;
use mmsb_proof::AdmissionProof;  // Canonical from mmsb-proof
use crate::dag::DependencyGraph;
pub(crate) fn commit_delta(
log: &TransactionLog,
admission_proof: &AdmissionProof,
delta: Delta,
active_dag: Option<&DependencyGraph>,
) -> std::io::Result<()> {
log.append(admission_proof, delta)
}
#[cfg(test)]
mod tests {
use super::commit_delta;
use crate::delta::Delta;
use crate::tlog::{self, TransactionLog};
use mmsb_proof::AdmissionProof;
use std::time::{SystemTime, UNIX_EPOCH};
fn base_delta() -> Delta {
// ... your existing base_delta ...
}
fn admission_proof(delta_hash: String) -> AdmissionProof {
AdmissionProof {
judgment_proof_hash: [0u8; 32],
epoch: 1,
nonce: 1,
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
let admission = admission_proof("bad-hash".to_string());
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
let admission = admission_proof(delta_hash);
commit_delta(&log, &admission, delta, None)
.expect("commit succeeds");
}
}
