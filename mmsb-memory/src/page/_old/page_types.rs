use thiserror::Error;
use mmsb_primitives::PageID;

/// Possible backing locations for a page.
///
/// This is a *semantic* enum and therefore does NOT belong in `mmsb-primitives`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    Cpu = 0,
    Gpu = 1,
    Unified = 2,
}

impl PageLocation {
    pub fn from_tag(tag: i32) -> Result<Self, PageError> {
        match tag {
            0 => Ok(PageLocation::Cpu),
            1 => Ok(PageLocation::Gpu),
            2 => Ok(PageLocation::Unified),
            other => Err(PageError::InvalidLocation(other)),
        }
    }
}

/// Errors related to page allocation, access, and decoding.
///
/// Note:
/// - `PageID` is a shared primitive and is *not* given a `Display` impl here
///   (foreign type + orphan rules).
/// - Errors should use `{:?}` for `PageID` formatting.
#[derive(Debug, Error)]
pub enum PageError {
    #[error("Invalid page size: {0}")]
    InvalidSize(usize),

    #[error("Invalid location tag: {0}")]
    InvalidLocation(i32),

    #[error("CUDA error code: {0}")]
    CudaError(i32),

    #[error("Page already freed")]
    AlreadyFreed,

    #[error("Invalid pointer")]
    InvalidPointer,

    #[error("Mask/payload size mismatch: expected {expected}, found {found}")]
    MaskSizeMismatch {
        expected: usize,
        found: usize,
    },

    #[error("PageID mismatch: expected {expected:?}, found {found:?}")]
    PageIDMismatch {
        expected: PageID,
        found: PageID,
    },

    #[error("Page not found: {0:?}")]
    PageNotFound(PageID),

    #[error("Page with ID {0:?} already exists")]
    AlreadyExists(PageID),

    #[error("Metadata decode error: {0}")]
    MetadataDecode(&'static str),

    #[error("Allocation failed")]
    AllocationFailed,

    #[error("Allocation error: code {0}")]
    AllocError(i32),
}
