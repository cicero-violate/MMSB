// Use the public prelude API
use mmsb_core::prelude::{EdgeType, ShadowPageGraph};
use mmsb_core::prelude::PageID;
use mmsb_core::prelude::ProvenanceTracker;
use std::sync::Arc;

#[test]
fn provenance_tracker_resolves_with_cache() {
    let graph = Arc::new(ShadowPageGraph::default());
    graph.add_edge(PageID(1), PageID(2), EdgeType::Data);
    graph.add_edge(PageID(2), PageID(3), EdgeType::Data);
    graph.add_edge(PageID(3), PageID(4), EdgeType::Data);
    let tracker = ProvenanceTracker::with_capacity(Arc::clone(&graph), 8, 8);
    let result = tracker.resolve(PageID(4));
    assert!(result.duration.as_millis() < 50);
    assert!(!result.chain.is_empty());
    let cached = tracker.resolve(PageID(4));
    assert!(cached.from_cache);
}
