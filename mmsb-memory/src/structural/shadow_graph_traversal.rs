use crate::structural::shadow_graph::ShadowPageGraph;
use mmsb_primitives::PageID;
use std::collections::{HashMap, VecDeque};

pub fn topological_sort(graph: &ShadowPageGraph) -> Vec<PageID> {
    // Simple BFS-based topsort using adjacency clones. For large graphs this should be replaced.
    let mut result = Vec::new();
    let mut in_degree = HashMap::new();
    let mut adjacency = HashMap::new();

    {
       // Build adjacency snapshots
       for (node, edges) in graph.adjacency.read().iter() {
            adjacency.insert(*node, edges.clone() as Vec<(PageID, crate::dag::EdgeType)>);
           in_degree.entry(*node).or_insert(0);
            for (child, _) in edges.iter() {
                *in_degree.entry(*child).or_insert(0) += 1;
            }
        }
    }

    let mut queue: VecDeque<PageID> = in_degree
        .iter()
        .filter_map(|(&node, &deg)| if deg == 0 { Some(node) } else { None })
        .collect();

    while let Some(node) = queue.pop_front() {
        result.push(node);
        if let Some(children) = adjacency.get(&node) {
            for (child, _) in children {
                if let Some(deg) = in_degree.get_mut(child) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(*child);
                    }
                }
            }
        }
    }

    result
}
