use super::allocator::{PageAllocator, PageSnapshotData};
use super::tlog::TransactionLog;
use crate::types::{PageID, PageLocation};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const SNAPSHOT_MAGIC: &[u8] = b"MMSBSNAP";
const SNAPSHOT_VERSION: u32 = 1;

pub fn write_checkpoint(
    allocator: &PageAllocator,
    tlog: &TransactionLog,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let pages = allocator.snapshot_pages();
    let log_offset = tlog.current_offset()?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(SNAPSHOT_MAGIC)?;
    writer.write_all(&SNAPSHOT_VERSION.to_le_bytes())?;
    writer.write_all(&(pages.len() as u32).to_le_bytes())?;
    writer.write_all(&log_offset.to_le_bytes())?;

    for page in pages {
        writer.write_all(&page.page_id.0.to_le_bytes())?;
        writer.write_all(&(page.size as u64).to_le_bytes())?;
        writer.write_all(&page.epoch.to_le_bytes())?;
        writer.write_all(&(page.location as i32).to_le_bytes())?;

        writer.write_all(&(page.metadata_blob.len() as u32).to_le_bytes())?;
        writer.write_all(&page.metadata_blob)?;

        writer.write_all(&(page.data.len() as u32).to_le_bytes())?;
        writer.write_all(&page.data)?;
    }
    writer.flush()?;
    Ok(())
}

pub fn load_checkpoint(
    allocator: &PageAllocator,
    tlog: &TransactionLog,
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != SNAPSHOT_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid checkpoint magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != SNAPSHOT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported checkpoint version",
        ));
    }

    let mut page_count_bytes = [0u8; 4];
    reader.read_exact(&mut page_count_bytes)?;
    let page_count = u32::from_le_bytes(page_count_bytes) as usize;

    let mut log_offset_bytes = [0u8; 8];
    reader.read_exact(&mut log_offset_bytes)?;
    let log_offset = u64::from_le_bytes(log_offset_bytes);

    let mut snapshots = Vec::with_capacity(page_count);
    for _ in 0..page_count {
        let mut id_bytes = [0u8; 8];
        reader.read_exact(&mut id_bytes)?;
        let id = PageID(u64::from_le_bytes(id_bytes));

        let mut size_bytes = [0u8; 8];
        reader.read_exact(&mut size_bytes)?;
        let size = u64::from_le_bytes(size_bytes) as usize;

        let mut epoch_bytes = [0u8; 4];
        reader.read_exact(&mut epoch_bytes)?;
        let epoch = u32::from_le_bytes(epoch_bytes);

        let mut loc_bytes = [0u8; 4];
        reader.read_exact(&mut loc_bytes)?;
        let location = PageLocation::from_tag(i32::from_le_bytes(loc_bytes)).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid location")
        })?;

        let mut metadata_len_bytes = [0u8; 4];
        reader.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u32::from_le_bytes(metadata_len_bytes) as usize;
        let mut metadata_blob = vec![0u8; metadata_len];
        reader.read_exact(&mut metadata_blob)?;

        let mut data_len_bytes = [0u8; 4];
        reader.read_exact(&mut data_len_bytes)?;
        let data_len = u32::from_le_bytes(data_len_bytes) as usize;
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        snapshots.push(PageSnapshotData {
            page_id: id,
            size,
            epoch,
            location,
            metadata_blob,
            data,
        });
    }

    allocator
        .restore_from_snapshot(snapshots)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;

    Ok(())
}
