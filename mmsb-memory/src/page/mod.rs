pub mod page_types;
pub mod page;
pub mod PageAllocator;
pub mod lockfree_allocator;

// Re-export public types from submodules — DO NOT include PageID here
pub use page_types::{PageError, PageLocation};
pub use page::{Page, Metadata};
pub use allocator::{PageAllocator, PageSnapshotData, PageAllocatorConfig};
pub use lockfree_allocator::LockFreeAllocator;

// Re-export Delta for convenience
pub use crate::delta::Delta;

// IMPORTANT: PageID is now exclusively from mmsb_primitives
// All files should import it directly: use mmsb_primitives::PageID;
// No re-export in mod.rs to avoid duplicate name/private import issues
