//! MemoryReader - Read-only query interface to MMSB

use mmsb_primitives::{Hash, PageID};

/// Read-only interface for querying canonical memory state
/// 
/// Services use this to:
/// - Check admission status
/// - Read committed facts
/// - Query page state
/// - Access deltas
pub trait MemoryReader: Send + Sync {
    /// Check if a judgment has been admitted
    fn check_admitted(&self, judgment_hash: Hash) -> bool;
    
    /// Get current epoch number
    fn current_epoch(&self) -> u64;
    
    /// Read a specific delta by hash
    fn get_delta(&self, delta_hash: Hash) -> Option<Vec<u8>>;
    
    /// Query page data by ID
    fn query_page(&self, page_id: PageID) -> Option<Vec<u8>>;
    
    /// Check if delta exists
    fn delta_exists(&self, delta_hash: Hash) -> bool;
}
