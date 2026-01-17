use super::page_types::PageID;
use crate::types::PageID;
use std::fmt;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaID(pub u64);

#[derive(Debug, Clone)]
pub struct Source(pub String);

#[derive(Debug, Clone)]
pub enum DeltaError {
    SizeMismatch { mask_len: usize, payload_len: usize },
    PageIDMismatch { expected: PageID, found: PageID },
    MaskSizeMismatch { expected: usize, found: usize },
}

impl fmt::Display for DeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeltaError::SizeMismatch { mask_len, payload_len } =>
                write!(f, "Mask/payload size mismatch mask={} payload={}", mask_len, payload_len),
            DeltaError::PageIDMismatch { expected, found } =>
                write!(f, "PageID mismatch: expected {:?}, found {:?}", expected, found),
            DeltaError::MaskSizeMismatch { expected, found } =>
                write!(f, "Mask size mismatch: expected {}, found {}", expected, found),
        }
    }
}

impl std::error::Error for DeltaError {}
