use crate::page::{Delta, DeltaID, Epoch, PageID, Source};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"MMSBLOG1";
const VERSION: u32 = 1;

#[derive(Debug)]
pub struct TransactionLog {
    entries: RwLock<VecDeque<Delta>>,
    writer: RwLock<Option<BufWriter<File>>>,
    path: PathBuf,
}

#[derive(Debug)]
pub struct TransactionLogReader {
    reader: BufReader<File>,
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

    pub fn append(&self, delta: Delta) -> std::io::Result<()> {
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
        validate_header(&mut reader)?;
        Ok(Self { reader })
    }

    pub fn next(&mut self) -> std::io::Result<Option<Delta>> {
        read_frame(&mut self.reader)
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
    validate_header(&mut reader)?;

    let mut summary = LogSummary::default();
    while let Ok(Some(delta)) = read_frame(&mut reader) {
        summary.total_deltas += 1;
        summary.total_bytes += delta.mask.len() as u64 + delta.payload.len() as u64 + 32;
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
    Ok(())
}

fn read_frame(reader: &mut BufReader<File>) -> std::io::Result<Option<Delta>> {
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

    Ok(Some(Delta {
        delta_id: DeltaID(u64::from_le_bytes(delta_id)),
        page_id: PageID(u64::from_le_bytes(page_id)),
        epoch: Epoch(u32::from_le_bytes(epoch)),
        mask,
        payload,
        is_sparse: sparse_flag[0] != 0,
        timestamp: u64::from_le_bytes(timestamp_bytes),
        source,
    }))
}

fn validate_header(reader: &mut BufReader<File>) -> std::io::Result<()> {
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
    if version != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported transaction log version",
        ));
    }
    Ok(())
}
