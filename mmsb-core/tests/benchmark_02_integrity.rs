// Use the public prelude API
use mmsb_core::prelude::{
    Delta, DeltaID, DeltaIntegrityChecker, DeviceBufferRegistry, Epoch, Page, PageID, PageLocation,
    Source,
};
use std::sync::Arc;

#[test]
fn integrity_checker_accepts_valid_delta() {
    let registry = Arc::new(DeviceBufferRegistry::default());
    let page = Arc::new(Page::new(PageID(1), 8, PageLocation::Cpu).unwrap());
    registry.insert(page);
    let delta = Delta {
        delta_id: DeltaID(1),
        page_id: PageID(1),
        epoch: Epoch(1),
        mask: vec![true; 8],
        payload: vec![1u8; 8],
        is_sparse: false,
        timestamp: 0,
        source: Source("test".into()),
        intent_metadata: None,
    };
    let mut checker = DeltaIntegrityChecker::new(Arc::clone(&registry));
    let report = checker.validate(&[delta]);
    assert!(report.passed());
}
