//! Cache locality optimizer using graph-based reordering

use std::collections::HashMap;

pub type PageId = u64;
pub type PhysAddr = u64;

/// Graph edge representing page dependency
#[derive(Debug, Clone)]
pub struct PageEdge {
    pub source: PageId,
    pub target: PageId,
    pub weight: f64,
}

/// Locality optimizer for cache-aware page placement
pub struct LocalityOptimizer {
    /// Page dependency graph
    edges: Vec<PageEdge>,
    /// Page size
    page_size: usize,
}

impl LocalityOptimizer {
    pub fn new(page_size: usize) -> Self {
        Self {
            edges: Vec::new(),
            page_size,
        }
    }

    pub fn add_edge(&mut self, source: PageId, target: PageId, weight: f64) {
        self.edges.push(PageEdge { source, target, weight });
    }

    /// Compute optimal page ordering using modified topological sort
    /// that respects locality (BFS-like traversal with weight prioritization)
    pub fn compute_ordering(&self) -> Vec<PageId> {
        if self.edges.is_empty() {
            return Vec::new();
        }

        // Build adjacency list
        let mut adj: HashMap<PageId, Vec<(PageId, f64)>> = HashMap::new();
        let mut all_pages: Vec<PageId> = Vec::new();
        
        for edge in &self.edges {
            adj.entry(edge.source).or_default().push((edge.target, edge.weight));
            if !all_pages.contains(&edge.source) {
                all_pages.push(edge.source);
            }
            if !all_pages.contains(&edge.target) {
                all_pages.push(edge.target);
            }
        }

        // Greedy ordering: start with highest-weight roots
        let mut ordered = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        // Sort pages by outgoing weight (heavy first)
        all_pages.sort_by(|a, b| {
            let weight_a: f64 = adj.get(a).map(|v| v.iter().map(|(_, w)| w).sum()).unwrap_or(0.0);
            let weight_b: f64 = adj.get(b).map(|v| v.iter().map(|(_, w)| w).sum()).unwrap_or(0.0);
            weight_b.partial_cmp(&weight_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &root in &all_pages {
            if visited.contains(&root) {
                continue;
            }
            Self::dfs_visit(root, &adj, &mut visited, &mut ordered);
        }

        ordered
    }

    fn dfs_visit(
        node: PageId,
        adj: &HashMap<PageId, Vec<(PageId, f64)>>,
        visited: &mut std::collections::HashSet<PageId>,
        ordered: &mut Vec<PageId>,
    ) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);

        if let Some(neighbors) = adj.get(&node) {
            let mut sorted_neighbors = neighbors.clone();
            sorted_neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            for (neighbor, _) in sorted_neighbors {
                Self::dfs_visit(neighbor, adj, visited, ordered);
            }
        }

        ordered.push(node);
    }

    /// Assign physical addresses based on ordering
    pub fn assign_addresses(&self, ordering: &[PageId]) -> HashMap<PageId, PhysAddr> {
        let mut placement = HashMap::new();
        for (i, &page_id) in ordering.iter().enumerate() {
            placement.insert(page_id, (i as u64) * (self.page_size as u64));
        }
        placement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_locality_optimizer() {
        let mut opt = LocalityOptimizer::new(4096);
        opt.add_edge(1, 2, 10.0);
        opt.add_edge(2, 3, 5.0);

        let ordering = opt.compute_ordering();
        assert_eq!(ordering.len(), 3);

        let addrs = opt.assign_addresses(&ordering);
        assert_eq!(addrs.len(), 3);
    }
}
