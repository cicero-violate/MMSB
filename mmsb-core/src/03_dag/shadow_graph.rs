use super::edge_types::EdgeType;
use crate::page::PageID;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub from: PageID,
    pub to: PageID,
    pub edge_type: EdgeType,
}

#[derive(Debug, Default)]
pub struct ShadowPageGraph {
    pub(crate) adjacency: RwLock<HashMap<PageID, Vec<(PageID, EdgeType)>>>,
}

impl ShadowPageGraph {
    pub fn add_edge(&self, from: PageID, to: PageID, edge_type: EdgeType) {
        let mut guard = self.adjacency.write();
        guard.entry(from).or_default().push((to, edge_type));
    }

    pub fn remove_edge(&self, from: PageID, to: PageID) {
        if let Some(edges) = self.adjacency.write().get_mut(&from) {
            edges.retain(|(target, _)| target != &to);
        }
    }

    pub fn descendants(&self, root: PageID) -> HashSet<PageID> {
        let graph = self.adjacency.read();
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                if let Some(children) = graph.get(&node) {
                    for (child, _) in children {
                        stack.push(*child);
                    }
                }
            }
        }
        seen
    }
}
