use mmsb_judgment::JudgmentToken;
use crate::page::{Delta, DeltaID, Epoch, PageID, Source};
use crate::utility::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

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

    /// Judgement boundary
    pub fn append(
        &self,
        token: &JudgmentToken,
        proof: &MmsbExecutionProof,
        delta: Delta,
    ) -> std::io::Result<()> {
        let _ = token;
        if proof.version != EXECUTION_PROOF_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "execution proof version mismatch",
            ));
        }
        let expected_hash = delta_hash(&delta);
        if proof.delta_hash != expected_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "execution proof hash mismatch",
            ));
        }
        {
            self.entries.write().push_back(delta.clone());
        }
        if let Some(writer) = self.writer.write().as_mut() {
            serialize_frame(writer, &delta)?;
            writer.flush()?;
            Ok(())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "transaction log writer closed",
            ))
        }
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    pub fn drain(&self) -> Vec<Delta> {
        let mut guard = self.entries.write();
        guard.drain(..).collect()
    }

    pub fn current_offset(&self) -> std::io::Result<u64> {
        let writer_lock = self.writer.read();
        if let Some(writer) = writer_lock.as_ref() {
            writer.get_ref().metadata().map(|meta| meta.len())
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "transaction log writer unavailable",
            ))
        }
    }

    // No truncate method needed — remove it if present
}

impl TransactionLogReader {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let version = validate_header(&mut reader)?;
        Ok(Self { reader, version })
    }

    pub fn next(&mut self) -> std::io::Result<Option<Delta>> {
        read_frame(&mut self.reader, self.version)
    }

    pub fn free(self) {}
}

impl Drop for TransactionLogReader {
    fn drop(&mut self) {}
}

pub fn summary(path: impl AsRef<Path>) -> std::io::Result<LogSummary> {
    let file = match File::open(path.as_ref()) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(err);
        }
        Err(err) => return Err(err),
    };
    if file.metadata()?.len() == 0 {
        return Ok(LogSummary::default());
    }
    let mut reader = BufReader::new(file);
    let version = validate_header(&mut reader)?;

    let mut summary = LogSummary::default();
    while let Ok(Some(delta)) = read_frame(&mut reader, version) {
        summary.total_deltas += 1;
        let metadata_bytes = delta
            .intent_metadata
            .as_ref()
            .map(|m| m.as_bytes().len() as u64)
            .unwrap_or(0);
        summary.total_bytes += delta.mask.len() as u64 + delta.payload.len() as u64 + 32 + metadata_bytes;
        summary.last_epoch = summary.last_epoch.max(delta.epoch.0);
    }
    Ok(summary)
}

fn serialize_frame(writer: &mut BufWriter<File>, delta: &Delta) -> std::io::Result<()> {
    writer.write_all(&delta.delta_id.0.to_le_bytes())?;
    writer.write_all(&delta.page_id.0.to_le_bytes())?;
    writer.write_all(&delta.epoch.0.to_le_bytes())?;

    let mask_len = delta.mask.len() as u32;
    writer.write_all(&mask_len.to_le_bytes())?;
    for flag in &delta.mask {
        writer.write_all(&[*flag as u8])?;
    }

    let payload_len = delta.payload.len() as u32;
    writer.write_all(&payload_len.to_le_bytes())?;
    writer.write_all(&delta.payload)?;

    writer.write_all(&[delta.is_sparse as u8])?;
    writer.write_all(&delta.timestamp.to_le_bytes())?;
    let source_bytes = delta.source.0.as_bytes();
    writer.write_all(&(source_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(source_bytes)?;
    let metadata_len = delta
        .intent_metadata
        .as_ref()
        .map(|s| s.as_bytes().len() as u32)
        .unwrap_or(0);
    writer.write_all(&metadata_len.to_le_bytes())?;
    if let Some(metadata) = &delta.intent_metadata {
        writer.write_all(metadata.as_bytes())?;
    }
    Ok(())
}

fn read_frame(reader: &mut BufReader<File>, version: u32) -> std::io::Result<Option<Delta>> {
    let mut delta_id = [0u8; 8];
    match reader.read_exact(&mut delta_id) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }

    let mut page_id = [0u8; 8];
    reader.read_exact(&mut page_id)?;
    let mut epoch = [0u8; 4];
    reader.read_exact(&mut epoch)?;

    let mut mask_len_bytes = [0u8; 4];
    reader.read_exact(&mut mask_len_bytes)?;
    let mask_len = u32::from_le_bytes(mask_len_bytes) as usize;
    let mut mask_raw = vec![0u8; mask_len];
    reader.read_exact(&mut mask_raw)?;
    let mask = mask_raw.iter().map(|b| *b != 0).collect::<Vec<bool>>();

    let mut payload_len_bytes = [0u8; 4];
    reader.read_exact(&mut payload_len_bytes)?;
    let payload_len = u32::from_le_bytes(payload_len_bytes) as usize;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;

    let mut sparse_flag = [0u8; 1];
    reader.read_exact(&mut sparse_flag)?;
    let mut timestamp_bytes = [0u8; 8];
    reader.read_exact(&mut timestamp_bytes)?;

    let mut source_len_bytes = [0u8; 4];
    reader.read_exact(&mut source_len_bytes)?;
    let source_len = u32::from_le_bytes(source_len_bytes) as usize;
    let mut source_buf = vec![0u8; source_len];
    reader.read_exact(&mut source_buf)?;
    let source = Source(String::from_utf8_lossy(&source_buf).to_string());

    let intent_metadata = if version >= 2 {
        let mut metadata_len_bytes = [0u8; 4];
        if reader.read_exact(&mut metadata_len_bytes).is_err() {
            return Ok(None);
        }
        let metadata_len = u32::from_le_bytes(metadata_len_bytes) as usize;
        if metadata_len == 0 {
            None
        } else {
            let mut metadata_buf = vec![0u8; metadata_len];
            reader.read_exact(&mut metadata_buf)?;
            Some(String::from_utf8_lossy(&metadata_buf).to_string())
        }
    } else {
        None
    };

    Ok(Some(Delta {
        delta_id: DeltaID(u64::from_le_bytes(delta_id)),
        page_id: PageID(u64::from_le_bytes(page_id)),
        epoch: Epoch(u32::from_le_bytes(epoch)),
        mask,
        payload,
        is_sparse: sparse_flag[0] != 0,
        timestamp: u64::from_le_bytes(timestamp_bytes),
        source,
        intent_metadata,
    }))
}

fn validate_header(reader: &mut BufReader<File>) -> std::io::Result<u32> {
    reader.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
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

pub(crate) fn delta_hash(delta: &Delta) -> String {
    let mut hasher = Sha256::new();
    hasher.update(delta.delta_id.0.to_le_bytes());
    hasher.update(delta.page_id.0.to_le_bytes());
    hasher.update(delta.epoch.0.to_le_bytes());
    hasher.update([delta.is_sparse as u8]);
    hasher.update(delta.timestamp.to_le_bytes());
    hasher.update(delta.mask.len().to_le_bytes());
    for flag in &delta.mask {
        hasher.update([*flag as u8]);
    }
    hasher.update(delta.payload.len().to_le_bytes());
    hasher.update(&delta.payload);
    let source_bytes = delta.source.0.as_bytes();
    hasher.update(source_bytes.len().to_le_bytes());
    hasher.update(source_bytes);
    if let Some(metadata) = &delta.intent_metadata {
        let meta_bytes = metadata.as_bytes();
        hasher.update(meta_bytes.len().to_le_bytes());
        hasher.update(meta_bytes);
    } else {
        hasher.update(0usize.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{delta_hash, TransactionLog};
    use crate::page::{Delta, DeltaID, Epoch, PageID, Source};
    use crate::utility::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
    use mmsb_judgment::JudgmentToken;
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
        let token = JudgmentToken::test_only();

        let err = log
            .append(&token, &execution, delta)
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
        let token = JudgmentToken::test_only();

        log.append(&token, &execution, delta)
            .expect("append succeeds");
    }
}
