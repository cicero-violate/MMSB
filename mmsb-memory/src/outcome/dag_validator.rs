use crate::dag::DependencyGraph;
use crate::structural::GraphValidationReport;
use mmsb_primitives::PageID;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub struct DagValidator<'a> {
    dag: &'a DependencyGraph,
}

impl<'a> DagValidator<'a> {
    pub fn new(dag: &'a DependencyGraph) -> Self {
        Self { dag }
    }

    pub fn detect_cycles(&self) -> GraphValidationReport {
        let start = Instant::now();
        let adjacency = self.dag.get_adjacency();
        let nodes: Vec<PageID> = adjacency.keys().copied().collect();
        
        let mut index = 0usize;
        let mut stack = Vec::new();
        let mut indices = HashMap::new();
        let mut lowlink = HashMap::new();
        let mut on_stack = HashSet::new();
        let mut cycle = Vec::new();
        let mut has_cycle = false;

        for node in nodes {
            if has_cycle {
                break;
            }
            if !indices.contains_key(&node) {
                strong_connect(
                    node,
                    adjacency,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlink,
                    &mut on_stack,
                    &mut cycle,
                    &mut has_cycle,
                );
            }
        }

        GraphValidationReport {
            has_cycle,
            cycle,
            visited: indices.len(),
            duration: start.elapsed(),
        }
    }
}

fn strong_connect(
    node: PageID,
    adjacency: &HashMap<PageID, Vec<(PageID, crate::dag::EdgeType)>>,
    index: &mut usize,
    stack: &mut Vec<PageID>,
    indices: &mut HashMap<PageID, usize>,
    lowlink: &mut HashMap<PageID, usize>,
    on_stack: &mut HashSet<PageID>,
    cycle: &mut Vec<PageID>,
    has_cycle: &mut bool,
) {
    let current_index = *index;
    indices.insert(node, current_index);
    lowlink.insert(node, current_index);
    *index += 1;
    stack.push(node);
    on_stack.insert(node);

    if let Some(neighbors) = adjacency.get(&node) {
        for (neighbor, _) in neighbors {
            if !indices.contains_key(neighbor) {
                strong_connect(
                    *neighbor,
                    adjacency,
                    index,
                    stack,
                    indices,
                    lowlink,
                    on_stack,
                    cycle,
                    has_cycle,
                );
                let neighbor_low = *lowlink.get(neighbor).unwrap();
                let node_low = *lowlink.get(&node).unwrap();
                lowlink.insert(node, node_low.min(neighbor_low));
            } else if on_stack.contains(neighbor) {
                let neighbor_idx = *indices.get(neighbor).unwrap();
                let node_low = *lowlink.get(&node).unwrap();
                lowlink.insert(node, node_low.min(neighbor_idx));
            }
        }
    }

    let node_low = *lowlink.get(&node).unwrap();
    let node_idx = *indices.get(&node).unwrap();
    if node_low == node_idx {
        let mut scc = Vec::new();
        loop {
            if let Some(w) = stack.pop() {
                on_stack.remove(&w);
                scc.push(w);
                if w == node {
                    break;
                }
            } else {
                break;
            }
        }
        if scc.len() > 1 && !*has_cycle {
            *has_cycle = true;
            *cycle = scc;
        }
    }
}
