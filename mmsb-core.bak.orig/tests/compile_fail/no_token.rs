use mmsb_core::prelude::{Delta, DeltaID, Epoch, PageID, Source, TransactionLog};

fn main() {
    let path = std::env::temp_dir().join("mmsb_no_token.tlog");
    let log = TransactionLog::new(path).unwrap();
    let delta = Delta {
        delta_id: DeltaID(1),
        page_id: PageID(1),
        epoch: Epoch(1),
        mask: vec![true; 1],
        payload: vec![0xAA],
        is_sparse: false,
        timestamp: 1,
        source: Source("no-token".to_string()),
        intent_metadata: None,
    };

    log.append(delta).unwrap();
}
