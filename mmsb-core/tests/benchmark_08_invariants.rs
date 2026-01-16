// Use the public prelude API
use mmsb_core::prelude::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
use mmsb_core::prelude::{InvariantChecker, InvariantContext};
use mmsb_core::dag::DependencyGraph;
use std::sync::Arc;

#[test]
fn invariant_checker_reports_success() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    allocator
        .allocate_raw(PageID(42), 256, Some(PageLocation::Cpu))
        .unwrap();
    let dag = DependencyGraph::new();
    let ctx = InvariantContext {
        allocator: Some(&allocator),
        graph: Some(&dag),
        registry: None,
    };
    let checker = InvariantChecker::with_builtins();
    for result in checker.run(&ctx) {
        assert!(result.passed);
    }
}
