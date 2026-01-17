// pub mod page_types;
// pub mod page;
// pub mod allocator;
// pub mod lockfree_allocator;

// pub use page_types::{PageError, PageID, PageLocation};
// pub use page::{Page, Metadata};
// pub use allocator::{PageAllocator, PageSnapshotData, PageAllocatorConfig};
// pub use lockfree_allocator::LockFreeAllocator;

// // Re-export Delta for convenience
// pub use crate::delta::Delta;

pub mod page_types;
pub mod page;
pub mod allocator;
pub mod lockfree_allocator;

// Re-export public types from submodules — AVOID re-exporting PageID multiple times
pub use page_types::{PageError, PageLocation};
pub use page::{Page, Metadata};
pub use allocator::{PageAllocator, PageSnapshotData, PageAllocatorConfig};
pub use lockfree_allocator::LockFreeAllocator;

// Re-export Delta for convenience
pub use crate::delta::Delta;

// IMPORTANT: Do NOT re-export PageID here — it is already available via mmsb_primitives
// Users should import PageID directly from mmsb_primitives or via page_types if needed
