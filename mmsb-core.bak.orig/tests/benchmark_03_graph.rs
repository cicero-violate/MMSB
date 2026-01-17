// Use the public prelude API
use mmsb_core::prelude::{EdgeType, StructuralOp};
use mmsb_core::dag::DagValidator;
use mmsb_core::prelude::PageID;

#[test]
fn graph_validator_detects_no_cycles() {
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
    ];
    let dag = mmsb_core::dag::build_dependency_graph(&ops);
    let validator = DagValidator::new(&dag);
    let report = validator.detect_cycles();
    assert!(!report.has_cycle);
    assert!(report.duration.as_millis() < 10);
}
