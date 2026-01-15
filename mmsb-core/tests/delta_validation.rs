// Use the public prelude API
use mmsb_core::prelude::{validate_delta, Delta, DeltaID, Epoch, PageID, Source};

fn dense_delta(payload: Vec<u8>, mask: Vec<bool>) -> Delta {
    Delta {
        delta_id: DeltaID(1),
        page_id: PageID(1),
        epoch: Epoch(0),
        mask,
        payload,
        is_sparse: false,
        timestamp: 0,
        source: Source("test".into()),
        intent_metadata: None,
    }
}

#[test]
fn validates_dense_lengths() {
    let delta = dense_delta(vec![1, 2, 3], vec![true, true, true]);
    assert!(validate_delta(&delta).is_ok());
}

#[test]
fn rejects_mismatched_dense_lengths() {
    let delta = dense_delta(vec![1, 2], vec![true, true, true]);
    assert!(validate_delta(&delta).is_err());
}
