use mmsb_core::prelude::{
    fold_add, Delta, DeltaID, Epoch, Page, PageID, PageLocation, PropagationCommand,
    PropagationEngine, Source, TransactionLog, TransactionLogReader, TropicalSemiring,
};
use mmsb_judgment::JudgmentToken;
use mmsb_core::prelude::utility::{EXECUTION_PROOF_VERSION, MmsbExecutionProof};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_log_path(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!("{}_{}.log", prefix, nanos));
    path
}

#[test]
fn integration_semiring_fold_delta_propagation_replay() {
    let semiring = TropicalSemiring;
    let values = vec![3.0, 1.5, 2.0];
    let folded = fold_add(&semiring, values);
    let payload = folded.to_le_bytes().to_vec();
    let mask = vec![true; payload.len()];

    let mut parent = Page::new(PageID(1), payload.len(), PageLocation::Cpu).unwrap();
    let delta = Delta::new_dense(
        DeltaID(1),
        parent.id,
        Epoch(1),
        payload.clone(),
        mask.clone(),
        Source("semiring".into()),
    )
    .unwrap();
    parent.apply_delta(&delta).unwrap();
    let parent_data = parent.data_slice().to_vec();

    let child = Page::new(PageID(2), payload.len(), PageLocation::Cpu).unwrap();
    let child_id = child.id;

    let log_path = temp_log_path("mmsb_semiring_prop_replay");
    let log = Arc::new(TransactionLog::new(&log_path).unwrap());
    let mask_for_child = Arc::new(mask);

    let engine = PropagationEngine::default();
    let log_for_cb = Arc::clone(&log);
    engine.register_callback(
        child_id,
        Arc::new(move |_, dependencies| {
            let parent = dependencies.first().expect("parent dependency");
            let payload = parent.data_slice().to_vec();
            let delta = Delta::new_dense(
                DeltaID(2),
                child_id,
                Epoch(2),
                payload,
                (*mask_for_child).clone(),
                Source("propagation".into()),
            )
            .expect("delta for propagated child");
            let delta_hash = delta_hash(&delta);
            let execution_proof = MmsbExecutionProof {
                version: EXECUTION_PROOF_VERSION,
                delta_hash,
                tool_call_id: "test".to_string(),
                tool_name: "test".to_string(),
                output: serde_json::json!({}),
                epoch: 0,
            };
            let token = JudgmentToken::test_only();
            log_for_cb
                .append(&token, &execution_proof, delta)
                .expect("append to log");
        }),
    );

    let parent_arc = Arc::new(parent);
    let child_arc = Arc::new(child);
    engine.enqueue(PropagationCommand {
        page_id: child_arc.id,
        page: Arc::clone(&child_arc),
        dependencies: vec![Arc::clone(&parent_arc)],
    });
    engine.drain();

    assert_eq!(log.len(), 1);

    let mut replay_page = Page::new(child_id, payload.len(), PageLocation::Cpu).unwrap();
    let mut reader = TransactionLogReader::open(&log_path).unwrap();
    while let Some(delta) = reader.next().unwrap() {
        delta.apply_to(&mut replay_page).unwrap();
    }

    assert_eq!(replay_page.data_slice(), parent_data.as_slice());

    let _ = std::fs::remove_file(&log_path);
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
