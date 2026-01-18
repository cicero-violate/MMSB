//! Replay Protection - Prevents duplicate execution admissions
//!
//! Maintains a record of all admitted executions to prevent replay attacks.

use mmsb_proof::Hash;
use parking_lot::RwLock;
use std::collections::HashSet;

/// Replay protection tracker
pub struct ReplayProtection {
    admitted: RwLock<HashSet<Hash>>,
}

impl ReplayProtection {
    pub fn new() -> Self {
        Self {
            admitted: RwLock::new(HashSet::new()),
        }
    }
    
    /// Check if an execution hash has already been admitted
    pub fn is_admitted(&self, hash: &Hash) -> bool {
        self.admitted.read().contains(hash)
    }
    
    /// Mark an execution hash as admitted
    pub fn mark_admitted(&self, hash: Hash) -> Result<(), ReplayError> {
        let mut admitted = self.admitted.write();
        if admitted.contains(&hash) {
            return Err(ReplayError::AlreadyAdmitted);
        }
        admitted.insert(hash);
        Ok(())
    }
    
    /// Clear all admitted hashes (for testing/replay scenarios)
    pub fn clear(&self) {
        self.admitted.write().clear();
    }
}

impl Default for ReplayProtection {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    AlreadyAdmitted,
}
