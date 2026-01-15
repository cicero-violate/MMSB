use crate::types::PageID;
use crate::dag::{EdgeType, StructuralOp};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
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
}
