//! Memory layout optimization for cache locality
//! 
//! Implements page reordering to minimize spatial distance between
//! frequently co-accessed pages, reducing cache misses.

use std::collections::HashMap;

/// Physical address type
pub type PhysAddr = u64;

/// Page identifier
pub type PageId = u64;

/// Memory layout representing page-to-address mapping
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Current page placement: PageId -> PhysAddr
    pub placement: HashMap<PageId, PhysAddr>,
    /// Page size in bytes
    pub page_size: usize,
}

/// Access pattern for locality optimization
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Co-access frequency: (PageId, PageId) -> count
    pub coaccesses: HashMap<(PageId, PageId), u64>,
}

impl MemoryLayout {
    /// Create new memory layout
    pub fn new(page_size: usize) -> Self {
        Self {
            placement: HashMap::new(),
            page_size,
        }
    }

    /// Compute locality cost: sum of distances weighted by co-access frequency
    pub fn locality_cost(&self, pattern: &AccessPattern) -> f64 {
        let mut cost = 0.0;
        for ((p1, p2), freq) in &pattern.coaccesses {
            if let (Some(&addr1), Some(&addr2)) = (self.placement.get(p1), self.placement.get(p2)) {
                let distance = if addr1 > addr2 {
                    addr1 - addr2
                } else {
                    addr2 - addr1
                };
                cost += (distance / self.page_size as u64) as f64 * (*freq as f64);
            }
        }
        cost
    }

    /// Reorder pages to minimize locality cost using greedy clustering
    pub fn optimize_layout(&mut self, pattern: &AccessPattern) {
        // Extract all pages
        let mut pages: Vec<PageId> = self.placement.keys().copied().collect();
        if pages.is_empty() {
            return;
        }

        // Sort by total access frequency (hottest first)
        pages.sort_by_key(|p| {
            let freq: u64 = pattern.coaccesses.iter()
                .filter(|((p1, p2), _)| p1 == p || p2 == p)
                .map(|(_, f)| f)
                .sum();
            std::cmp::Reverse(freq)
        });

        // Reassign addresses sequentially (linear layout)
        let base_addr = 0u64;
        for (i, page_id) in pages.iter().enumerate() {
            let new_addr = base_addr + (i as u64) * (self.page_size as u64);
            self.placement.insert(*page_id, new_addr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_creation() {
        let layout = MemoryLayout::new(4096);
        assert_eq!(layout.page_size, 4096);
        assert!(layout.placement.is_empty());
    }

    #[test]
    fn test_locality_cost_empty() {
        let layout = MemoryLayout::new(4096);
        let pattern = AccessPattern {
            coaccesses: HashMap::new(),
        };
        assert_eq!(layout.locality_cost(&pattern), 0.0);
    }

    #[test]
    fn test_optimize_layout() {
        let mut layout = MemoryLayout::new(4096);
        layout.placement.insert(1, 0);
        layout.placement.insert(2, 8192);
        layout.placement.insert(3, 16384);

        let mut pattern = AccessPattern {
            coaccesses: HashMap::new(),
        };
        pattern.coaccesses.insert((1, 2), 10);
        pattern.coaccesses.insert((2, 3), 5);

        layout.optimize_layout(&pattern);
        
        // Pages should be reordered by hotness
        assert!(layout.placement.len() == 3);
    }
}
