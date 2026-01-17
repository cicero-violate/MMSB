use crate::dag::{DependencyGraph, EdgeType};
use crate::structural::StructuralOp;
use mmsb_primitives::PageID;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"MMSBDAGL";
const VERSION: u32 = 1;
const ENV_LOG_PATH: &str = "MMSB_DAG_LOG_PATH";
const DEFAULT_LOG_NAME: &str = "mmsb_dag.log";

pub fn append_structural_record(
    path: impl AsRef<Path>,
    ops: &[StructuralOp],
) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path.as_ref())?;

    ensure_log_header(&mut file)?;

    write_u64(&mut file, ops.len() as u64)?;
    for op in ops {
        match op {
            StructuralOp::AddEdge { from, to, edge_type } => {
                file.write_all(&[0])?;
                write_u64(&mut file, from.0)?;
                write_u64(&mut file, to.0)?;
                write_u8(&mut file, *edge_type as u8)?;  // Fixed cast
            }
            StructuralOp::RemoveEdge { from, to } => {
                file.write_all(&[1])?;
                write_u64(&mut file, from.0)?;
                write_u64(&mut file, to.0)?;
            }
        }
    }
    file.flush()?;

    Ok(())
}

pub fn default_structural_log_path() -> PathBuf {
    std::env::var(ENV_LOG_PATH)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LOG_NAME))
}

fn ensure_log_header(file: &mut File) -> io::Result<()> {
    let mut header = [0u8; MAGIC.len() + 4];
    let size = file.metadata()?.len();
    if size == 0 {
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.flush()?;
    } else {
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header)?;
        if &header[0..MAGIC.len()] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid log magic",
            ));
        }
        let version = u32::from_le_bytes(header[MAGIC.len()..].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported log version",
            ));
        }
    }
    Ok(())
}

fn write_u64(file: &mut File, value: u64) -> io::Result<()> {
    file.write_all(&value.to_le_bytes())
}

fn write_u8(file: &mut File, value: u8) -> io::Result<()> {
    file.write_all(&[value])
}

fn read_u64(file: &mut File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u8(file: &mut File) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    file.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn replay_structural_log(path: impl AsRef<Path>) -> io::Result<DependencyGraph> {
    let mut file = File::open(path.as_ref())?;
    let mut dag = DependencyGraph::new();

    let mut header = [0u8; MAGIC.len() + 4];
    file.read_exact(&mut header)?;
    if &header[0..MAGIC.len()] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid log magic",
        ));
    }
    let version = u32::from_le_bytes(header[MAGIC.len()..].try_into().unwrap());
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported log version",
        ));
    }

    while let Ok(len) = read_u64(&mut file) {
        for _ in 0..len {
            let op_type = read_u8(&mut file)?;
            match op_type {
                0 => {
                    let from = PageID(read_u64(&mut file)?);
                    let to = PageID(read_u64(&mut file)?);
                    let edge_type_raw = read_u8(&mut file)?;
                    let edge_type = EdgeType::try_from(edge_type_raw).unwrap_or(EdgeType::Data); // Add try_from if needed
                    dag.add_edge(from, to, edge_type);  // Assume method exists or add it
                }
                1 => {
                    let from = PageID(read_u64(&mut file)?);
                    let to = PageID(read_u64(&mut file)?);
                    dag.remove_edge(from, to);  // Assume method exists or add it
                }
                other => return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid op type: {}", other),
                )),
            }
        }
    }

    Ok(dag)
}

#[cfg(test)]
mod tests {
    use super::{append_structural_record, default_structural_log_path, replay_structural_log};
    use crate::structural::{EdgeType, StructuralOp};
    use mmsb_primitives::PageID;
    use std::env;
    use std::time::SystemTime;

    #[test]
    fn test_append_and_replay_structural_log() {
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_path = env::temp_dir().join(format!("test_dag_log_{nanos}.bin"));
        let ops = vec![
            StructuralOp::AddEdge {
                from: PageID(1),
                to: PageID(2),
                edge_type: EdgeType::Data,
            },
            StructuralOp::RemoveEdge {
                from: PageID(2),
                to: PageID(3),
            },
        ];

        append_structural_record(&temp_path, &ops).expect("append failed");
        let dag = replay_structural_log(&temp_path).expect("replay failed");

        assert_eq!(dag.edges().len(), 1);
        let _ = std::fs::remove_file(&temp_path);
    }
}
