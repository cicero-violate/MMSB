use crate::dag::DependencyGraph;
use std::path::Path;
use std::io::{self, Write, Read};
use std::fs::{File, OpenOptions};

const MAGIC: &[u8] = b"MMSBDAG1";
const VERSION: u32 = 1;

pub fn write_dag_snapshot(
    dag: &DependencyGraph,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;

    file.write_all(MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;

    let dag_version = dag.version();
    file.write_all(&dag_version.to_le_bytes())?;

    let edges = dag.edges();
    let edge_count = edges.len() as u64;
    file.write_all(&edge_count.to_le_bytes())?;

    for (from, to, edge_type) in edges {
        file.write_all(&from.0.to_le_bytes())?;
        file.write_all(&to.0.to_le_bytes())?;
        let edge_type_byte = edge_type_to_byte(edge_type);
        file.write_all(&[edge_type_byte])?;
    }

    file.flush()?;
    Ok(())
}

pub fn load_dag_snapshot(path: impl AsRef<Path>) -> io::Result<DependencyGraph> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid DAG snapshot magic",
        ));
    }

    let mut version_bytes = [0u8; 4];
    file.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported DAG snapshot version: {}", version),
        ));
    }

    let mut dag_version_bytes = [0u8; 8];
    file.read_exact(&mut dag_version_bytes)?;
    let _dag_version = u64::from_le_bytes(dag_version_bytes);

    let mut edge_count_bytes = [0u8; 8];
    file.read_exact(&mut edge_count_bytes)?;
    let edge_count = u64::from_le_bytes(edge_count_bytes);

    use crate::types::PageID;
    use crate::dag::StructuralOp;
    let mut ops = Vec::new();

    for _ in 0..edge_count {
        let mut from_bytes = [0u8; 8];
        let mut to_bytes = [0u8; 8];
        let mut edge_type_byte = [0u8; 1];

        file.read_exact(&mut from_bytes)?;
        file.read_exact(&mut to_bytes)?;
        file.read_exact(&mut edge_type_byte)?;

        let from = PageID(u64::from_le_bytes(from_bytes));
        let to = PageID(u64::from_le_bytes(to_bytes));
        let edge_type = byte_to_edge_type(edge_type_byte[0])?;

        ops.push(StructuralOp::AddEdge { from, to, edge_type });
    }

    Ok(crate::dag::build_dependency_graph(&ops))
}

fn edge_type_to_byte(edge_type: crate::dag::EdgeType) -> u8 {
    use crate::dag::EdgeType;
    match edge_type {
        EdgeType::Data => 0,
        EdgeType::Control => 1,
        EdgeType::Gpu => 2,
        EdgeType::Compiler => 3,
    }
}

fn byte_to_edge_type(byte: u8) -> io::Result<crate::dag::EdgeType> {
    use crate::dag::EdgeType;
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
    use crate::types::PageID;
    use crate::dag::{StructuralOp, EdgeType};
    use std::env;

    #[test]
    fn snapshot_roundtrip() {
        let ops = vec![
            StructuralOp::AddEdge {
                from: PageID(1),
                to: PageID(2),
                edge_type: EdgeType::Data,
            },
            StructuralOp::AddEdge {
                from: PageID(2),
                to: PageID(3),
                edge_type: EdgeType::Data,
            },
        ];
        let dag = crate::dag::build_dependency_graph(&ops);

        let temp_path = env::temp_dir().join("test_dag_snapshot.bin");
        write_dag_snapshot(&dag, &temp_path).expect("write failed");

        let loaded = load_dag_snapshot(&temp_path).expect("load failed");
        assert_eq!(loaded.edges().len(), 2);
    }
}
