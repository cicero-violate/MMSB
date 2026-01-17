use std::fmt;
use crate::types::{PageID, PageLocation, PageError};

// Page-specific metadata (not canonical types)
#[derive(Debug, Clone)]
pub struct PageMetadata {
    pub size: usize,
    pub location: PageLocation,
}
