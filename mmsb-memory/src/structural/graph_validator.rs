use crate::structural::shadow_graph::ShadowPageGraph;
use crate::page::PageID;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct GraphValidationReport {
    pub has_cycle: bool,
    pub cycle: Vec<PageID>,
    pub visited: usize,
    pub duration: Duration,
}

pub(crate) struct GraphValidator<'a> {
    graph: &'a ShadowPageGraph,
}

impl<'a> GraphValidator<'a> {
    pub(crate) fn new(graph: &'a ShadowPageGraph) -> Self {
        Self { graph }
    }

    pub(crate) fn detect_cycles(&self) -> GraphValidationReport {
        self.run(None)
    }

    pub(crate) fn validate_page(&self, root: PageID) -> GraphValidationReport {
        self.run(Some(root))
    }

    fn run(&self, root: Option<PageID>) -> GraphValidationReport {
        let start = Instant::now();
        let adjacency = self.graph.adjacency.read().clone();
        let nodes: Vec<PageID> = match root {
            Some(root) => reachable(&adjacency, root),
            None => adjacency.keys().copied().collect(),
        };
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
                    &adjacency,
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

fn reachable(
    adjacency: &HashMap<PageID, Vec<(PageID, crate::dag::edge_types::EdgeType)>>,
    root: PageID,
) -> Vec<PageID> {
    let mut visited = HashSet::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            if let Some(children) = adjacency.get(&node) {
                for (child, _) in children {
                    stack.push(*child);
                }
            }
        }
    }
    visited.into_iter().collect()
}

fn strong_connect(
    node: PageID,
    adjacency: &HashMap<PageID, Vec<(PageID, crate::dag::edge_types::EdgeType)>>,
    index: &mut usize,
    stack: &mut Vec<PageID>,
    indices: &mut HashMap<PageID, usize>,
    lowlink: &mut HashMap<PageID, usize>,
    on_stack: &mut HashSet<PageID>,
    cycle: &mut Vec<PageID>,
    has_cycle: &mut bool,
) {
    indices.insert(node, *index);
    lowlink.insert(node, *index);
    *index += 1;
    stack.push(node);
    on_stack.insert(node);

    if let Some(children) = adjacency.get(&node) {
        for (child, _) in children {
            if !indices.contains_key(child) {
                strong_connect(
                    *child,
                    adjacency,
                    index,
                    stack,
                    indices,
                    lowlink,
                    on_stack,
                    cycle,
                    has_cycle,
                );
                let child_low = *lowlink.get(child).unwrap();
                let node_low = lowlink.get_mut(&node).unwrap();
                *node_low = (*node_low).min(child_low);
            } else if on_stack.contains(child) {
                let child_index = *indices.get(child).unwrap();
                let node_low = lowlink.get_mut(&node).unwrap();
                *node_low = (*node_low).min(child_index);
            }
            if *has_cycle {
                return;
            }
        }
    }

    let node_low = *lowlink.get(&node).unwrap();
    let node_index = *indices.get(&node).unwrap();
    if node_low == node_index {
        let mut component = Vec::new();
        loop {
            let popped = stack.pop().unwrap();
            on_stack.remove(&popped);
            component.push(popped);
            if popped == node {
                break;
            }
        }
        if component.len() > 1 || is_self_loop(adjacency, node) {
            *cycle = component;
            *has_cycle = true;
        }
    }
}

fn is_self_loop(
    adjacency: &HashMap<PageID, Vec<(PageID, crate::dag::edge_types::EdgeType)>>,
    node: PageID,
) -> bool {
    adjacency
        .get(&node)
        .map(|edges| edges.iter().any(|(target, _)| *target == node))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::GraphValidator;
    use crate::dag::{build_dependency_graph, EdgeType, StructuralOp};
    use crate::dag::shadow_graph::ShadowPageGraph;
    use crate::page::PageID;

    #[test]
    fn detects_cycle() {
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
                to: PageID(1),
                edge_type: EdgeType::Data,
            },
        ];
        let dag = build_dependency_graph(&ops);
        let graph = ShadowPageGraph::from_dag_and_ops(&dag, &[]);
        let validator = GraphValidator::new(&graph);
        let report = validator.detect_cycles();
        assert!(report.has_cycle);
        assert!(report.cycle.len() >= 3);
    }

    #[test]
    fn per_page_validation() {
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
        ];
        let dag = build_dependency_graph(&ops);
        let graph = ShadowPageGraph::from_dag_and_ops(&dag, &[]);
        let validator = GraphValidator::new(&graph);
        let report = validator.validate_page(PageID(1));
        assert!(!report.has_cycle);
        assert!(report.duration.as_micros() > 0);
    }
}
