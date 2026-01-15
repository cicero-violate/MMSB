use mmsb_core::prelude::*;

#[derive(Debug, PartialEq)]
struct PageSnapshot {
    id: PageID,
    location: PageLocation,
    epoch: Epoch,
    data: Vec<u8>,
    mask: Vec<u8>,
    metadata: Vec<u8>,
}

fn snapshot_page(page: &Page) -> PageSnapshot {
    PageSnapshot {
        id: page.id,
        location: page.location(),
        epoch: page.epoch(),
        data: page.data_slice().to_vec(),
        mask: page.mask_slice().to_vec(),
        metadata: page.metadata_blob(),
    }
}

fn assert_invariants_hold(ctx: &InvariantContext<'_>, checker: &InvariantChecker) {
    let results = checker.run(ctx);
    let failures: Vec<String> = results
        .iter()
        .filter(|result| !result.passed)
        .map(|result| {
            format!(
                "{}: {}",
                result.name,
                result.details.as_deref().unwrap_or("no details")
            )
        })
        .collect();
    assert!(failures.is_empty(), "invariant failures: {failures:?}");
}

#[test]
fn passthrough_preserves_invariants_and_state() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let src_ptr = allocator
        .allocate_raw(PageID(1), 128, Some(PageLocation::Cpu))
        .expect("source allocation failed");
    let dst_ptr = allocator
        .allocate_raw(PageID(2), 128, Some(PageLocation::Cpu))
        .expect("target allocation failed");

    unsafe {
        let src = &mut *src_ptr;
        let dst = &mut *dst_ptr;
        src.data_mut_slice().fill(0x11);
        dst.data_mut_slice().fill(0x22);
        src.set_epoch(Epoch(1));
        dst.set_epoch(Epoch(2));
        src.set_metadata(vec![("source".to_string(), vec![1, 2, 3])]);
        dst.set_metadata(vec![("target".to_string(), vec![4, 5, 6])]);
    }

    let graph = ShadowPageGraph::default();
    let ctx = InvariantContext {
        allocator: Some(&allocator),
        graph: Some(&graph),
        registry: None,
    };
    let checker = InvariantChecker::with_builtins();
    assert_invariants_hold(&ctx, &checker);

    let src_before = unsafe { snapshot_page(&*src_ptr) };
    let dst_before = unsafe { snapshot_page(&*dst_ptr) };

    unsafe {
        passthrough(&*src_ptr, &mut *dst_ptr);
    }

    let src_after = unsafe { snapshot_page(&*src_ptr) };
    let dst_after = unsafe { snapshot_page(&*dst_ptr) };

    assert_eq!(src_after, src_before);
    assert_eq!(dst_after, dst_before);
    assert_invariants_hold(&ctx, &checker);
}
