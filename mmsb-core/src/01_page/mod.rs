#![allow(unused_imports)]

pub mod allocator;
pub mod columnar_delta;
pub mod delta;
pub mod device;
pub mod device_registry;
pub mod epoch;
pub mod host_device_sync;
pub mod integrity_checker;
pub mod lockfree_allocator;
pub mod page;
pub mod replay_validator;

pub mod checkpoint;
pub mod delta_merge;
pub mod delta_validation;
pub mod simd_mask;
pub mod tlog;
pub mod tlog_compression;
pub mod tlog_replay;
pub mod tlog_serialization;
mod page_commit;

use mmsb_judgment::JudgmentToken;
use crate::utility::{MmsbAdmissionProof, MmsbExecutionProof};
use crate::dag::DependencyGraph;

pub use crate::types::{PageID, PageLocation, PageError, Epoch, EpochCell, DeltaID, Source, DeltaError};
pub use allocator::{PageAllocator, PageAllocatorConfig, PageInfo, PageSnapshotData};
pub use columnar_delta::ColumnarDeltaBatch;
pub use checkpoint::{load_checkpoint, write_checkpoint};
pub use delta::Delta;
pub use delta_merge::merge_deltas;
pub use delta_validation::validate_delta;
pub use device::DeviceRegistry;
pub use device_registry::DeviceBufferRegistry;
pub use host_device_sync::HostDeviceSync;
pub use integrity_checker::{DeltaIntegrityChecker, IntegrityReport, IntegrityViolation, IntegrityViolationKind};
pub use lockfree_allocator::LockFreeAllocator;
pub use page::{Metadata, Page};
pub use replay_validator::{ReplayCheckpoint, ReplayReport, ReplayValidator};
pub use simd_mask::generate_mask;
pub use tlog::{LogSummary, TransactionLog, TransactionLogReader, summary};
pub use tlog_compression::compact;
pub use tlog_replay::apply_log;
pub use tlog_serialization::read_log;

pub(crate) fn commit_delta(
    log: &TransactionLog,
    token: &JudgmentToken,
    admission_proof: &MmsbAdmissionProof,
    execution_proof: &MmsbExecutionProof,
    delta: Delta,
    active_dag: Option<&DependencyGraph>,
) -> std::io::Result<()> {
    page_commit::commit_delta(log, token, admission_proof, execution_proof, delta, active_dag)
}
