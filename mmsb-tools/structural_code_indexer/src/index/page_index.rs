//! Page Index with Stable PageIDs
//!
//! **CRITICAL:** PageID = hash(canonical_path)
//! Must be deterministic across runs.

use mmsb_core::types::PageID;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Page index mapping paths to stable PageIDs
#[derive(Debug, Clone)]
pub struct PageIndex {
    path_to_id: HashMap<PathBuf, PageID>,
    id_to_path: HashMap<PageID, PathBuf>,
}

impl PageIndex {
    pub fn new() -> Self {
        Self {
            path_to_id: HashMap::new(),
            id_to_path: HashMap::new(),
        }
    }
    
    /// Get or create stable PageID for a path
    ///
    /// **Deterministic:** Same path always returns same PageID
    pub fn get_or_create_page(&mut self, path: &Path) -> PageID {
        let canonical = canonicalize_path(path);
        
        if let Some(&page_id) = self.path_to_id.get(&canonical) {
            return page_id;
        }
        
        let page_id = path_to_page_id(&canonical);
        self.path_to_id.insert(canonical.clone(), page_id);
        self.id_to_path.insert(page_id, canonical);
        
        page_id
    }
    
    /// Get PageID for a path if it exists
    pub fn get_page(&self, path: &Path) -> Option<PageID> {
        let canonical = canonicalize_path(path);
        self.path_to_id.get(&canonical).copied()
    }
    
    /// Get path for a PageID
    pub fn get_path(&self, page_id: PageID) -> Option<&Path> {
        self.id_to_path.get(&page_id).map(|p: &PathBuf| p.as_path())
    }
    
    /// Get all pages
    pub fn all_pages(&self) -> Vec<PageID> {
        self.id_to_path.keys().copied().collect()
    }
}

impl Default for PageIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert path to stable PageID via hash
///
/// **CRITICAL:** This must be deterministic
fn path_to_page_id(path: &Path) -> PageID {
    let path_str = path.to_string_lossy();
    let mut hasher = Sha256::new();
    hasher.update(path_str.as_bytes());
    let hash = hasher.finalize();
    
    // Take first 8 bytes as u64
    let id = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ]);
    
    PageID(id)
}

/// Canonicalize path (make relative to repo root)
fn canonicalize_path(path: &Path) -> PathBuf {
    // Strip common prefixes and normalize
    path.components()
        .filter(|c| !matches!(c, std::path::Component::CurDir | std::path::Component::ParentDir))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_deterministic_page_id() {
        let path = Path::new("src/main.rs");
        let id1 = path_to_page_id(path);
        let id2 = path_to_page_id(path);
        
        assert_eq!(id1, id2);
    }
    
    #[test]
    fn test_different_paths_different_ids() {
        let path1 = Path::new("src/main.rs");
        let path2 = Path::new("src/lib.rs");
        
        let id1 = path_to_page_id(path1);
        let id2 = path_to_page_id(path2);
        
        assert_ne!(id1, id2);
    }
    
    #[test]
    fn test_page_index_stable() {
        let mut index = PageIndex::new();
        
        let path = Path::new("src/main.rs");
        let id1 = index.get_or_create_page(path);
        let id2 = index.get_or_create_page(path);
        
        assert_eq!(id1, id2);
        assert_eq!(index.get_path(id1), Some(path));
    }
}
