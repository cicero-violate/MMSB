use crate::dag::DependencyGraph;
use mmsb_primitives::PageID;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ProvenanceResult {
    pub chain: Vec<PageID>,
    pub duration: std::time::Duration,
    pub from_cache: bool,
}

pub struct ProvenanceTracker {
    graph: Arc<DependencyGraph>,
    cache: parking_lot::Mutex<HashMap<PageID, Vec<PageID>>>,
    order: parking_lot::Mutex<VecDeque<PageID>>,
    capacity: usize,
    depth_limit: usize,
}

impl ProvenanceTracker {
    pub fn new(graph: Arc<DependencyGraph>) -> Self {
        Self::with_capacity(graph, 128, 32)
    }

    pub fn with_capacity(
        graph: Arc<DependencyGraph>,
        capacity: usize,
        depth_limit: usize,
    ) -> Self {
        Self {
            graph,
            cache: parking_lot::Mutex::new(HashMap::new()),
            order: parking_lot::Mutex::new(VecDeque::new()),
            capacity: capacity.max(4),
            depth_limit: depth_limit.max(1),
        }
    }

    pub fn resolve(&self, page_id: PageID) -> ProvenanceResult {
        let start = Instant::now();
        if let Some(chain) = self.cache.lock().get(&page_id).cloned() {
            return ProvenanceResult {
                chain,
                duration: start.elapsed(),
                from_cache: true,
            };
        }
        let chain = self.resolve_uncached(page_id);
        self.insert_cache(page_id, chain.clone());
        ProvenanceResult {
            chain,
            duration: start.elapsed(),
            from_cache: false,
        }
    }

    fn resolve_uncached(&self, page_id: PageID) -> Vec<PageID> {
        let mut reverse: HashMap<PageID, Vec<PageID>> = HashMap::new();
        for (from, edges) in self.graph.get_adjacency().iter() {
            for (to, _) in edges {
                reverse.entry(*to).or_default().push(*from);
            }
        }
        let mut chain = Vec::new();
        let mut current = page_id;
        chain.push(current);
        for _ in 0..self.depth_limit {
            if let Some(parents) = reverse.get(&current) {
                if let Some(parent) = parents.first() {
                    current = *parent;
                    chain.push(current);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        chain
    }

    fn insert_cache(&self, key: PageID, chain: Vec<PageID>) {
        let mut cache = self.cache.lock();
        let mut order = self.order.lock();
        if cache.contains_key(&key) {
            order.retain(|id| id != &key);
        }
        cache.insert(key, chain);
        order.push_front(key);
        while order.len() > self.capacity {
            if let Some(evicted) = order.pop_back() {
                cache.remove(&evicted);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProvenanceTracker;
    use crate::dag::{build_dependency_graph, EdgeType, StructuralOp};
    use crate::page::PageID;
    use std::sync::Arc;

    #[test]
    fn resolves_chain_with_depth_limit() {
        let ops = vec![
            StructuralOp::AddEdge {
                from: PageID(1),
                to: PageID(2),
                edge_type: EdgeType::Data,
            },
            StructuralOp::AddEdge {
                from: PageID(2),
                to: PageID(3),
                edge_type: EdgeType::Data,
            },
            StructuralOp::AddEdge {
                from: PageID(3),
                to: PageID(4),
                edge_type: EdgeType::Data,
            },
        ];
        let dag = Arc::new(build_dependency_graph(&ops));
        let tracker = ProvenanceTracker::with_capacity(Arc::clone(&dag), 16, 2);
        let result = tracker.resolve(PageID(4));
        assert_eq!(result.chain.len(), 3); // 4 -> 3 -> 2 (depth limit)
        assert!(!result.from_cache);
        let cached = tracker.resolve(PageID(4));
        assert!(cached.from_cache);
    }

    #[test]
    fn cache_does_not_grow_unbounded() {
        let ops: Vec<StructuralOp> = (1..=10)
            .map(|id| StructuralOp::AddEdge {
                from: PageID(id),
                to: PageID(id + 1),
                edge_type: EdgeType::Data,
            })
            .collect();
        let dag = Arc::new(build_dependency_graph(&ops));
        let tracker = ProvenanceTracker::with_capacity(Arc::clone(&dag), 4, 4);
        for id in 5..=10 {
            tracker.resolve(PageID(id));
        }
        assert!(tracker.cache.lock().len() <= 4);
    }
}
