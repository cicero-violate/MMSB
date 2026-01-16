// Use the public prelude API
use mmsb_core::prelude::{EdgeType, PageID, ProvenanceTracker, StructuralOp};
use mmsb_core::dag::build_dependency_graph;
use std::sync::Arc;

#[test]
fn provenance_tracker_resolves_with_cache() {
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
        StructuralOp::AddEdge {
            from: PageID(3),
            to: PageID(4),
            edge_type: EdgeType::Data,
        },
    ];
    let dag = Arc::new(build_dependency_graph(&ops));
    let tracker = ProvenanceTracker::with_capacity(Arc::clone(&dag), 8, 8);
    let result = tracker.resolve(PageID(4));
    assert!(result.duration.as_millis() < 50);
    assert!(!result.chain.is_empty());
    let cached = tracker.resolve(PageID(4));
    assert!(cached.from_cache);
}
