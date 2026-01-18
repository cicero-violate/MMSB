//! PageView - Non-owning projection of Page for device operations
//!
//! Authority: NONE
//! Ownership: NONE
//! Purpose: Expose minimal device-needed data without allocation/Drop responsibilities

use mmsb_primitives::PageID;
use crate::page::page_types::PageLocation;

/// Non-owning view of a Page for device/kernel operations
/// 
/// Invariants:
/// - Does not own memory (no Drop)
/// - Contains only read-only facts + raw pointers
/// - No epoch, metadata, or allocation authority
#[derive(Debug, Clone, Copy)]
pub struct PageView {
    pub id: PageID,
    pub location: PageLocation,
    pub data: *mut u8,
    pub mask: *mut u8,
    pub len: usize,
}

impl PageView {
    /// Safe read-only access to data slice
    #[inline]
    pub fn data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    /// Safe read-only access to mask slice
    #[inline]
    pub fn mask_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.mask, (self.len + 7) / 8) }
    }
}

// PageView is Send/Sync because it's just POD + raw pointers
// Safety: User must ensure the pointed-to memory remains valid
unsafe impl Send for PageView {}
unsafe impl Sync for PageView {}
