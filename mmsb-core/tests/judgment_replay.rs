use mmsb_core::prelude::{
    Delta, DeltaID, Epoch, Page, PageID, PageLocation, Source, TransactionLog,
    TransactionLogReader,
};
use mmsb_core::prelude::utility::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
use mmsb_judgment::issue::issue_judgment;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn deterministic_replay_with_judgment() -> std::io::Result<()> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mmsb_judgment_replay_{nanos}.tlog"));
    let log = TransactionLog::new(&path)?;

    let payload = vec![0x10, 0x20, 0x30, 0x40];
    let mask = vec![true; payload.len()];
    let delta = Delta::new_dense(
        DeltaID(1),
        PageID(1),
        Epoch(1),
        payload.clone(),
        mask,
        Source("judgment-replay".to_string()),
    )
    .expect("valid delta");

    let mut committed = Page::new(PageID(1), payload.len(), PageLocation::Cpu)
        .expect("committed page");
    committed.apply_delta(&delta).expect("apply delta");
    let committed_snapshot = committed.data_slice().to_vec();

    let delta_hash = delta_hash(&delta);
    let execution_proof = MmsbExecutionProof {
        version: EXECUTION_PROOF_VERSION,
        delta_hash: delta_hash.clone(),
        tool_call_id: "test".to_string(),
        tool_name: "test".to_string(),
        output: serde_json::json!({}),
        epoch: 0,
    };
    let token = issue_judgment("deterministic replay", &delta_hash);
    log.append(&token, &execution_proof, delta)?;

    let mut replayed = Page::new(PageID(1), payload.len(), PageLocation::Cpu)
        .expect("replay page");
    let mut reader = TransactionLogReader::open(&path)?;
    while let Some(delta) = reader.next()? {
        delta.apply_to(&mut replayed).expect("replay apply");
    }

    assert_eq!(committed_snapshot, replayed.data_slice());

    let _ = std::fs::remove_file(&path);
    Ok(())
}

fn delta_hash(delta: &Delta) -> String {
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
