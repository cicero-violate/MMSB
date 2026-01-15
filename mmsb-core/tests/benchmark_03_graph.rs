// Use the public prelude API
use mmsb_core::prelude::{EdgeType, GraphValidator, ShadowPageGraph};
use mmsb_core::prelude::PageID;

#[test]
fn graph_validator_detects_no_cycles() {
    let graph = ShadowPageGraph::default();
    graph.add_edge(PageID(1), PageID(2), EdgeType::Data);
    graph.add_edge(PageID(2), PageID(3), EdgeType::Data);
    let validator = GraphValidator::new(&graph);
    let report = validator.detect_cycles();
    assert!(!report.has_cycle);
    assert!(report.duration.as_millis() < 10);
}
