use mmsb_core::dag::ShadowPageGraph;
use mmsb_core::page::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
use mmsb_core::utility::{InvariantChecker, InvariantContext};
use std::sync::Arc;

#[test]
fn invariant_checker_reports_success() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    allocator
        .allocate_raw(PageID(42), 256, Some(PageLocation::Cpu))
        .unwrap();
    let graph = ShadowPageGraph::default();
    let ctx = InvariantContext {
        allocator: Some(&allocator),
        graph: Some(&graph),
        registry: None,
    };
    let checker = InvariantChecker::with_builtins();
    for result in checker.run(&ctx) {
        assert!(result.passed);
    }
}
