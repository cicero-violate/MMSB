use mmsb_primitives::PageID;
use crate::dag::EdgeType;
use crate::structural::StructuralOp;
use std::collections::{HashMap, HashSet};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DependencyGraph {
    adjacency: HashMap<PageID, Vec<(PageID, EdgeType)>>,
    version: u64,
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            version: 0,
        }
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn descendants(&self, root: PageID) -> HashSet<PageID> {
        let mut seen = HashSet::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if seen.insert(node) {
                if let Some(children) = self.adjacency.get(&node) {
                    for (child, _) in children {
                        stack.push(*child);
                    }
                }
            }
        }
        seen
    }

    pub fn edges(&self) -> Vec<(PageID, PageID, EdgeType)> {
        let mut result = Vec::new();
        for (from, targets) in &self.adjacency {
            for (to, edge_type) in targets {
                result.push((*from, *to, *edge_type));
            }
        }
        result
    }

    pub fn has_edge(&self, from: PageID, to: PageID) -> bool {
        self.adjacency
            .get(&from)
            .map(|targets| targets.iter().any(|(t, _)| *t == to))
            .unwrap_or(false)
    }

    pub fn contains_page(&self, page_id: PageID) -> bool {
        self.adjacency.contains_key(&page_id)
            || self.adjacency.values().any(|targets| {
                targets.iter().any(|(t, _)| *t == page_id)
            })
    }

    pub fn add_edge(&mut self, from: PageID, to: PageID, edge_type: EdgeType) {
        self.adjacency
            .entry(from)
            .or_default()
            .push((to, edge_type));
        self.version += 1;
    }

    pub fn remove_edge(&mut self, from: PageID, to: PageID) {
        if let Some(edges) = self.adjacency.get_mut(&from) {
            edges.retain(|(target, _)| *target != to);
        }
        self.version += 1;
    }


    pub(crate) fn apply_ops(&mut self, ops: &[StructuralOp]) {
        for op in ops {
            match op {
                StructuralOp::AddEdge { from, to, edge_type } => {
                    self.adjacency
                        .entry(*from)
                        .or_default()
                        .push((*to, *edge_type));
                }
                StructuralOp::RemoveEdge { from, to } => {
                    if let Some(edges) = self.adjacency.get_mut(from) {
                        edges.retain(|(target, _)| target != to);
                    }
                }
            }
        }
        self.version += 1;
    }

    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    pub fn get_adjacency(&self) -> &HashMap<PageID, Vec<(PageID, EdgeType)>> {
        &self.adjacency
    }

    pub fn compute_snapshot_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.version.to_le_bytes());
        
        let mut edges = self.edges();
        edges.sort_by_key(|(from, to, _)| (from.0, to.0));
        
        for (from, to, edge_type) in edges {
            hasher.update(&from.0.to_le_bytes());
            hasher.update(&to.0.to_le_bytes());
            let et = format!("{:?}", edge_type);
            hasher.update(et.as_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }
}
