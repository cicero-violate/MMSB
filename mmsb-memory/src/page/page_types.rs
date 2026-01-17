use std::fmt;

/// Possible backing locations for a page
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

/// Globally unique identifier for pages
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PageID(pub u64);

impl fmt::Display for PageID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub enum PageError {
    InvalidSize(usize),
    InvalidLocation(i32),
    CudaError(i32),
    AlreadyFreed,
    InvalidPointer,
}

impl fmt::Display for PageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PageError::InvalidSize(s) => write!(f, "Invalid page size: {}", s),
            PageError::InvalidLocation(t) => write!(f, "Invalid location tag: {}", t),
            PageError::CudaError(c) => write!(f, "CUDA error code: {}", c),
            PageError::AlreadyFreed => write!(f, "Page already freed"),
            PageError::InvalidPointer => write!(f, "Invalid pointer"),
        }
    }
}

impl std::error::Error for PageError {}
