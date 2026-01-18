use crate::dag::{DependencyGraph, EdgeType};
use crate::structural::StructuralOp;
use crate::proofs::MmsbStructuralAdmissionProof;
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
    proof: &MmsbStructuralAdmissionProof,
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
                file.write_all(&[edge_type_to_byte(*edge_type)])?;
            }
            StructuralOp::RemoveEdge { from, to } => {
                file.write_all(&[1])?;
                write_u64(&mut file, from.0)?;
                write_u64(&mut file, to.0)?;
                file.write_all(&[0])?;
            }
        }
    }

    write_u32(&mut file, proof.version)?;
    file.write_all(&[proof.approved as u8])?;
    write_u64(&mut file, proof.epoch)?;
    write_string(&mut file, &proof.ops_hash)?;
    write_optional_string(&mut file, proof.dag_snapshot_hash.as_deref())?;
    write_string(&mut file, &proof.conversation_id)?;
    write_string(&mut file, &proof.message_id)?;
    write_string(&mut file, &proof.scope)?;

    file.flush()?;
    file.sync_all()?;
    Ok(())
}

pub fn replay_structural_log(path: impl AsRef<Path>) -> io::Result<DependencyGraph> {
    let path = path.as_ref();
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            return Ok(DependencyGraph::new());
        }
        Err(err) => return Err(err),
    };

    read_log_header(&mut file)?;

    let mut all_ops = Vec::new();
    loop {
        let ops_len = match read_u64_opt(&mut file)? {
            Some(value) => value,
            None => break,
        };

        let mut ops = Vec::with_capacity(ops_len as usize);
        for _ in 0..ops_len {
            let mut op_tag = [0u8; 1];
            file.read_exact(&mut op_tag)?;
            let from = PageID(read_u64(&mut file)?);
            let to = PageID(read_u64(&mut file)?);
            let mut edge_type_byte = [0u8; 1];
            file.read_exact(&mut edge_type_byte)?;

            let op = match op_tag[0] {
                0 => StructuralOp::AddEdge {
                    from,
                    to,
                    edge_type: byte_to_edge_type(edge_type_byte[0])?,
                },
                1 => StructuralOp::RemoveEdge { from, to },
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown structural op tag: {}", op_tag[0]),
                    ));
                }
            };
            ops.push(op);
        }

        // Proof metadata (stored for audit; ignored for replay execution)
        let _proof_version = read_u32(&mut file)?;
        let mut approved = [0u8; 1];
        file.read_exact(&mut approved)?;
        let _epoch = read_u64(&mut file)?;
        let _ops_hash = read_string(&mut file)?;
        let _dag_snapshot_hash = read_optional_string(&mut file)?;
        let _conversation_id = read_string(&mut file)?;
        let _message_id = read_string(&mut file)?;
        let _scope = read_string(&mut file)?;

        all_ops.extend(ops);
    }

    Ok(crate::commit::build_dependency_graph(&all_ops))
}

pub fn default_structural_log_path() -> io::Result<PathBuf> {
    if let Ok(path) = std::env::var(ENV_LOG_PATH) {
        return Ok(PathBuf::from(path));
    }
    if cfg!(test) {
        let mut path = std::env::temp_dir();
        path.push("mmsb_dag_test.log");
        return Ok(path);
    }
    let mut path = std::env::current_dir()?;
    path.push(DEFAULT_LOG_NAME);
    Ok(path)
}

fn ensure_log_header(file: &mut File) -> io::Result<()> {
    let len = file.metadata()?.len();
    if len == 0 {
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        return Ok(());
    }
    validate_log_header_for_append(file)
}

fn read_log_header(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid DAG log magic",
        ));
    }
    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported DAG log version: {}", version),
        ));
    }
    Ok(())
}

fn validate_log_header_for_append(file: &mut File) -> io::Result<()> {
    read_log_header(file)?;
    file.seek(SeekFrom::End(0))?;
    Ok(())
}

fn write_u64(file: &mut File, value: u64) -> io::Result<()> {
    file.write_all(&value.to_le_bytes())
}

fn write_u32(file: &mut File, value: u32) -> io::Result<()> {
    file.write_all(&value.to_le_bytes())
}

fn write_string(file: &mut File, value: &str) -> io::Result<()> {
    write_u64(file, value.len() as u64)?;
    file.write_all(value.as_bytes())
}

fn write_optional_string(file: &mut File, value: Option<&str>) -> io::Result<()> {
    match value {
        Some(value) => {
            file.write_all(&[1])?;
            write_string(file, value)
        }
        None => file.write_all(&[0]),
    }
}

fn read_u64(file: &mut File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_u64_opt(file: &mut File) -> io::Result<Option<u64>> {
    let mut buf = [0u8; 8];
    match file.read_exact(&mut buf) {
        Ok(()) => Ok(Some(u64::from_le_bytes(buf))),
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(err) => Err(err),
    }
}

fn read_u32(file: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_string(file: &mut File) -> io::Result<String> {
    let len = read_u64(file)? as usize;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 string")
    })?)
}

fn read_optional_string(file: &mut File) -> io::Result<Option<String>> {
    let mut flag = [0u8; 1];
    file.read_exact(&mut flag)?;
    if flag[0] == 1 {
        Ok(Some(read_string(file)?))
    } else if flag[0] == 0 {
        Ok(None)
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid optional string flag: {}", flag[0]),
        ))
    }
}

fn edge_type_to_byte(edge_type: EdgeType) -> u8 {
    match edge_type {
        EdgeType::Data => 0,
        EdgeType::Control => 1,
        EdgeType::Gpu => 2,
        EdgeType::Compiler => 3,
    }
}

fn byte_to_edge_type(byte: u8) -> io::Result<EdgeType> {
    match byte {
        0 => Ok(EdgeType::Data),
        1 => Ok(EdgeType::Control),
        2 => Ok(EdgeType::Gpu),
        3 => Ok(EdgeType::Compiler),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown edge type byte: {}", byte),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::MmsbStructuralAdmissionProof;
    use crate::page::PageID;
    use std::env;

    #[test]
    fn replay_log_roundtrip() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
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
        let proof = MmsbStructuralAdmissionProof::new(
            &ops,
            None,
            "conv".to_string(),
            "msg".to_string(),
            "scope".to_string(),
            true,
            1,
        );

        append_structural_record(&temp_path, &ops, &proof).expect("append failed");
        let dag = replay_structural_log(&temp_path).expect("replay failed");

        assert_eq!(dag.edges().len(), 1);
        let _ = std::fs::remove_file(&temp_path);
    }
}
