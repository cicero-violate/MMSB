use mmsb_core::prelude::{Delta, DeltaID, Epoch, PageID, Source, TransactionLog};
use mmsb_core::prelude::proof::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
use mmsb_judgment::issue::issue_judgment;
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn judgment_issued_token_allows_single_commit() -> std::io::Result<()> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mmsb_judgment_integration_{nanos}.tlog"));
    let log = TransactionLog::new(&path)?;

    let delta = Delta {
        delta_id: DeltaID(1),
        page_id: PageID(1),
        epoch: Epoch(1),
        mask: vec![true; 1],
        payload: vec![0xAB],
        is_sparse: false,
        timestamp: 1,
        source: Source("judgment-integration".to_string()),
        intent_metadata: Some("commit after judgment".to_string()),
    };

    let delta_hash = delta_hash(&delta);
    let execution_proof = MmsbExecutionProof {
        version: EXECUTION_PROOF_VERSION,
        delta_hash: delta_hash.clone(),
        tool_call_id: "test".to_string(),
        tool_name: "test".to_string(),
        output: serde_json::json!({}),
        epoch: 0,
    };
    let token = issue_judgment("commit after judgment", &delta_hash);
    log.append(&token, &execution_proof, delta)?;

    assert_eq!(log.len(), 1);

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
