//! Hot page clustering for improved cache utilization

use std::collections::{HashMap, HashSet};

pub type PageId = u64;

/// Cluster of pages that should be co-located
#[derive(Debug, Clone)]
pub struct PageCluster {
    /// Pages in this cluster
    pub pages: HashSet<PageId>,
    /// Total access frequency
    pub hotness: u64,
}

/// Page clustering engine
#[derive(Debug)]
pub struct PageClusterer {
    /// Current clusters
    clusters: Vec<PageCluster>,
    /// Minimum cluster size
    min_cluster_size: usize,
}

impl PageClusterer {
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            clusters: Vec::new(),
            min_cluster_size,
        }
    }

    /// Cluster pages based on co-access patterns
    pub fn cluster_pages(&mut self, coaccesses: &HashMap<(PageId, PageId), u64>) {
        self.clusters.clear();

        // Build affinity matrix
        let mut affinities: HashMap<PageId, HashMap<PageId, u64>> = HashMap::new();
        for ((p1, p2), freq) in coaccesses {
            affinities.entry(*p1).or_default().insert(*p2, *freq);
            affinities.entry(*p2).or_default().insert(*p1, *freq);
        }

        // Greedy clustering: start with hottest pages
        let mut unclustered: HashSet<PageId> = affinities.keys().copied().collect();
        
        while !unclustered.is_empty() {
            // Find hottest unclustered page
            let seed = unclustered.iter()
                .max_by_key(|&p| {
                    affinities.get(p).map(|adj| adj.values().sum::<u64>()).unwrap_or(0)
                })
                .copied()
                .unwrap();

            // Build cluster around seed
            let mut cluster = HashSet::new();
            cluster.insert(seed);
            unclustered.remove(&seed);

            // Add neighbors with strong affinity
            if let Some(neighbors) = affinities.get(&seed) {
                let mut candidates: Vec<_> = neighbors.iter()
                    .filter(|(p, _)| unclustered.contains(p))
                    .collect();
                candidates.sort_by_key(|(_, freq)| std::cmp::Reverse(**freq));

                for (&neighbor, _) in candidates.iter().take(self.min_cluster_size - 1) {
                    cluster.insert(neighbor);
                    unclustered.remove(&neighbor);
                }
            }

            let hotness = cluster.iter()
                .filter_map(|p| affinities.get(p))
                .flat_map(|adj| adj.values())
                .sum();

            self.clusters.push(PageCluster { pages: cluster, hotness });
        }

        // Sort clusters by hotness (hottest first)
        self.clusters.sort_by_key(|c| std::cmp::Reverse(c.hotness));
    }

    pub fn clusters(&self) -> &[PageCluster] {
        &self.clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_clustering() {
        let mut clusterer = PageClusterer::new(2);
        let mut coaccesses = HashMap::new();
        coaccesses.insert((1, 2), 100);
        coaccesses.insert((2, 3), 50);
        coaccesses.insert((4, 5), 80);

        clusterer.cluster_pages(&coaccesses);

        assert!(clusterer.clusters().len() > 0);
        assert!(clusterer.clusters()[0].hotness > 0);
    }
}
