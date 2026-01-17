//! Centralized type definitions for mmsb-memory
//! Canonical type spine - ALL shared identifiers live here

use std::fmt;
use serde::{Serialize, Deserialize};

// ===== Core Identifiers =====

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageID(pub u64);

impl fmt::Display for PageID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeltaID(pub u64);

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Epoch(pub u64);

impl Epoch {
    pub fn next(&self) -> Self {
        Epoch(self.0 + 1)
    }
}

// ===== Enums =====

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    Cpu = 0,
    Gpu = 1,
    Unified = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeType {
    Data = 0,
    Control = 1,
    Gpu = 2,
    Compiler = 3,
}

#[derive(Debug, Clone)]
pub struct Source(pub String);

// ===== Errors =====

#[derive(Debug, Clone)]
pub enum PageError {
    InvalidSize(usize),
    InvalidLocation(i32),
    AlreadyFreed,
    InvalidPointer,
}

impl fmt::Display for PageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageError::InvalidSize(s) => write!(f, "Invalid page size: {}", s),
            PageError::InvalidLocation(t) => write!(f, "Invalid location tag: {}", t),
            PageError::AlreadyFreed => write!(f, "Page already freed"),
            PageError::InvalidPointer => write!(f, "Invalid pointer"),
        }
    }
}

impl std::error::Error for PageError {}

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
