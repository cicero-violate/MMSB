use crate::page::PageID;
use thiserror::Error;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaID(pub u64);

#[derive(Debug, Clone)]
pub struct Source(pub String);

#[derive(Debug, Error)]
pub enum DeltaError {
    #[error("Mask/payload size mismatch mask={mask_len} payload={payload_len}")]
    SizeMismatch { mask_len: usize, payload_len: usize },

    #[error("PageID mismatch: expected {expected:?}, found {found:?}")]
    PageIDMismatch { expected: PageID, found: PageID },

    #[error("Mask size mismatch: expected {expected}, found {found}")]
    MaskSizeMismatch { expected: usize, found: usize },
}
