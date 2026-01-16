//! Propagation Statistics Collection (Read-Only)
//!
//! Observes system behavior to inform Phase 7 proposals.
//! **Never mutates state.**

use crate::types::PageID;
use std::collections::HashMap;

/// Observed propagation behavior over a time window
#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    /// Fanout count per page (how many downstream pages affected)
    pub fanout_per_page: HashMap<PageID, usize>,
    
    /// Derived delta count per page (how many times page was re-derived)
    pub derived_delta_count: HashMap<PageID, usize>,
    
    /// Total propagation events observed
    pub total_propagations: usize,
    
    /// Observation window (epoch range)
    pub epoch_range: Option<(u64, u64)>,
}

impl PropagationStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record a propagation event
    ///
    /// **NOTE:** This is for Phase 7 observation only.
    /// Does not affect Phase 4 (propagation execution).
    pub fn record_propagation(&mut self, root: PageID, affected: &[PageID]) {
        *self.fanout_per_page.entry(root).or_insert(0) += affected.len();
        
        for page in affected {
            *self.derived_delta_count.entry(*page).or_insert(0) += 1;
        }
        
        self.total_propagations += 1;
    }
    
    /// Get pages with fanout above threshold
    pub fn high_fanout_pages(&self, threshold: usize) -> Vec<(PageID, usize)> {
        self.fanout_per_page
            .iter()
            .filter(|(_, &count)| count > threshold)
            .map(|(&page, &count)| (page, count))
            .collect()
    }
    
    /// Get pages with no observed fanout (dead dependencies candidate)
    pub fn zero_fanout_pages(&self) -> Vec<PageID> {
        self.fanout_per_page
            .iter()
            .filter(|(_, &count)| count == 0)
            .map(|(&page, _)| page)
            .collect()
    }
    
    /// Get median fanout
    pub fn median_fanout(&self) -> Option<usize> {
        if self.fanout_per_page.is_empty() {
            return None;
        }
        
        let mut counts: Vec<usize> = self.fanout_per_page.values().copied().collect();
        counts.sort_unstable();
        Some(counts[counts.len() / 2])
    }
    
    /// Get pages re-derived more than threshold times
    pub fn frequently_derived_pages(&self, threshold: usize) -> Vec<(PageID, usize)> {
        self.derived_delta_count
            .iter()
            .filter(|(_, &count)| count > threshold)
            .map(|(&page, &count)| (page, count))
            .collect()
    }
}
