use crate::diff::RepoDiff;
use crate::error::CodeEditError;
use mmsb_core::prelude::Delta;
use mmsb_core::types::{DeltaID, Epoch, PageID, Source};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct MappedEdit {
    pub deltas: Vec<Delta>,
    pub ops: Vec<mmsb_core::dag::StructuralOp>,
}

pub fn map_edit(diff: &RepoDiff) -> Result<MappedEdit, CodeEditError> {
    let mut deltas = Vec::new();
    for (idx, change) in diff.file_changes.iter().enumerate() {
        let payload = change.content.clone().unwrap_or_default();
        let mask = vec![true; payload.len()];
        let path_text = change.path.to_string_lossy();
        let delta_id = DeltaID(hash_u64(path_text.as_bytes(), &payload));
        let page_id = PageID(hash_u64(path_text.as_bytes(), b""));
        let delta = Delta {
            delta_id,
            page_id,
            epoch: Epoch(idx as u32),
            mask,
            payload,
            is_sparse: false,
            timestamp: idx as u64,
            source: Source(path_text.to_string()),
            intent_metadata: None,
        };
        deltas.push(delta);
    }

    Ok(MappedEdit {
        deltas,
        ops: diff.structural_ops.clone(),
    })
}

fn hash_u64(prefix: &[u8], payload: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(payload);
    let hash = hasher.finalize();
    u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ])
}
