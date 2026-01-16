//! Symbol Index
//!
//! Maps symbols to PageIDs for reference resolution.

use mmsb_core::types::PageID;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Symbol index for resolving imports
#[derive(Debug, Clone)]
pub struct SymbolIndex {
    /// symbol → PageID
    symbol_to_page: HashMap<String, PageID>,
}

impl SymbolIndex {
    pub fn new() -> Self {
        Self {
            symbol_to_page: HashMap::new(),
        }
    }
    
    /// Register a symbol defined in a page
    pub fn register_symbol(&mut self, symbol: String, page_id: PageID) {
        self.symbol_to_page.insert(symbol, page_id);
    }
    
    /// Resolve a symbol reference to PageID
    ///
    /// Returns None if symbol cannot be resolved (which is acceptable)
    pub fn resolve_symbol(&self, symbol: &str) -> Option<PageID> {
        self.symbol_to_page.get(symbol).copied()
    }
    
    /// Attempt to resolve a module reference to a file path
    ///
    /// Best-effort resolution. Falls back to file-level edges.
    pub fn resolve_mod_to_path(&self, base_path: &Path, mod_name: &str) -> Option<PathBuf> {
        let base_dir = base_path.parent()?;
        
        // Try inline mod: src/foo.rs + mod bar → src/foo/bar.rs
        let _inline_path = base_dir.join(mod_name).with_extension("rs");
        
        // Try adjacent mod: src/lib.rs + mod foo → src/foo.rs
        let adjacent_path = base_dir.join(format!("{}.rs", mod_name));
        
        // Return first that seems reasonable
        // In real implementation, we'd check filesystem
        Some(adjacent_path)
    }
}

impl Default for SymbolIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_register_and_resolve() {
        let mut index = SymbolIndex::new();
        let page_id = PageID(42);
        
        index.register_symbol("foo::Bar".to_string(), page_id);
        
        assert_eq!(index.resolve_symbol("foo::Bar"), Some(page_id));
        assert_eq!(index.resolve_symbol("unknown"), None);
    }
}
