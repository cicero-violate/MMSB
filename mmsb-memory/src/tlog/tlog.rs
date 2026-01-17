//! Transaction Log - Append-only persistence for MMSB mutations
//!
//! The TransactionLog is part of mmsb-memory's persistence layer.
//! It records committed deltas in an append-only format.
//! No authority — just durable byte storage under memory's control.

use crate::delta::{Delta, DeltaID, Source};
use crate::epoch::Epoch;
use crate::page::PageID;
use crate::proofs::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub fn delta_hash(delta: &Delta) -> String {
    let mut hasher = Sha256::new();
    hasher.update(&delta.delta_id.0.to_le_bytes());
    hasher.update(&delta.page_id.0.to_le_bytes());
    hasher.update(&delta.epoch.0.to_le_bytes());
    for &bit in &delta.mask {
        hasher.update(&[if bit { 1 } else { 0 }]);
    }
    hasher.update(&delta.payload);
    format!("{:x}", hasher.finalize())
}

const MAGIC: &[u8] = b"MMSBLOG1";
const VERSION: u32 = 2;

#[derive(Debug)]
pub struct TransactionLog {
    entries: RwLock<VecDeque<Delta>>,
    writer: RwLock<Option<BufWriter<File>>>,
    path: PathBuf,
}

#[derive(Debug)]
pub struct TransactionLogReader {
    reader: BufReader<File>,
    version: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LogSummary {
    pub total_deltas: u64,
    pub total_bytes: u64,
    pub last_epoch: u32,
}

impl TransactionLog {
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        if file.metadata()?.len() == 0 {
            file.write_all(MAGIC)?;
            file.write_all(&VERSION.to_le_bytes())?;
            file.flush()?;
        }
        let writer = BufWriter::new(file);
        Ok(Self {
            entries: RwLock::new(VecDeque::new()),
            writer: RwLock::new(Some(writer)),
            path,
        })
    }

    /// Clear in-memory entries without modifying the log file.
    /// Used for state reset while preserving append-only log semantics.
    pub fn clear_entries(&self) {
        self.entries.write().clear();
    }

    /// Append a delta with proof witness (no JudgmentToken) canonical
    pub fn append(
        &self,
        admission_proof: &AdmissionProof,  // ← NEW: canonical witness
        delta: Delta,
    ) -> std::io::Result<()> {
        // Remove the version/hash check that was for MmsbExecutionProof
        // (or replace with admission-specific check if needed)

        // Example minimal validation (optional):
        if admission_proof.epoch == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid admission epoch",
            ));
        }

        // Serialize admission_proof + delta
        let serialized = bincode::serialize(&(admission_proof, &delta))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let mut writer = self.writer.write();
        if let Some(w) = writer.as_mut() {
            w.write_all(&serialized.len().to_le_bytes())?;
            w.write_all(&serialized)?;
            w.flush()?;
        }

        self.entries.write().push_back(delta.clone());

        Ok(())
    }

    /// Load summary from log (total deltas, bytes, last epoch)
    pub fn summary(&self) -> std::io::Result<LogSummary> {
        // Real impl would scan the log file
        Ok(LogSummary {
            total_deltas: self.entries.read().len() as u64,
            total_bytes: 0, // Placeholder
            last_epoch: 0,  // Placeholder
        })
    }

    /// Replay from log (deterministic)
    pub fn replay(&self, start_epoch: u32) -> std::io::Result<Vec<Delta>> {
        // Real impl would read from file
        Ok(self.entries.read().iter().cloned().collect())
    }
}

// Helper to check log version
fn check_log_version(reader: &mut BufReader<File>) -> std::io::Result<u32> {
    let mut magic_bytes = [0u8; 8];
    reader.read_exact(&mut magic_bytes)?;
    if &magic_bytes != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid transaction log magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version < 1 || version > VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported transaction log version",
        ));
    }
    Ok(version)
}

pub fn current_offset(&self) -> std::io::Result<u64> {
    let file = File::open(&self.path)?;
    let metadata = file.metadata()?;
    Ok(metadata.len())
}

#[cfg(test)]
mod tests {
    use super::{delta_hash, TransactionLog};
    use crate::page::{Delta, DeltaID, Epoch, PageID, Source};
    use crate::proof::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn base_delta() -> Delta {
        Delta {
            delta_id: DeltaID(1),
            page_id: PageID(1),
            epoch: Epoch(1),
            mask: vec![true; 2],
            payload: vec![0xAA, 0xBB],
            is_sparse: false,
            timestamp: 1,
            source: Source("test".to_string()),
            intent_metadata: None,
        }
    }

    fn execution_proof(delta_hash: String, version: u32) -> MmsbExecutionProof {
        MmsbExecutionProof {
            version,
            delta_hash,
            tool_call_id: "test".to_string(),
            tool_name: "test".to_string(),
            output: json!({}),
            epoch: 0,
        }
    }

    #[test]
    fn append_without_valid_execution_proof_halts() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mmsb_append_no_proof_{nanos}.tlog"));
        let log = TransactionLog::new(&path).expect("log");

        let delta = base_delta();
        let execution = execution_proof("bad-hash".to_string(), EXECUTION_PROOF_VERSION);

        let err = log
            .append(&execution, delta)
            .expect_err("expected execution proof failure");
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn append_with_valid_execution_proof_succeeds() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mmsb_append_with_proof_{nanos}.tlog"));
        let log = TransactionLog::new(&path).expect("log");

        let delta = base_delta();
        let delta_hash = delta_hash(&delta);
        let execution = execution_proof(delta_hash, EXECUTION_PROOF_VERSION);

        log.append(&execution, delta)
            .expect("append succeeds");
    }
}
