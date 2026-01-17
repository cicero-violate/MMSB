use super::shadow_graph::ShadowPageGraph;
use crate::types::PageID;
use std::collections::HashMap;

enum VisitState {
    Visiting,
    Visited,
}
/// Detect whether the given graph contains a cycle using DFS.
pub fn has_cycle(graph: &ShadowPageGraph) -> bool {
    let adjacency = graph.adjacency.read().clone();
    let mut states: HashMap<PageID, VisitState> = HashMap::new();
    fn dfs(
        node: PageID,
        adjacency: &HashMap<PageID, Vec<(PageID, crate::dag::edge_types::EdgeType)>>,
        states: &mut HashMap<PageID, VisitState>,
    ) -> bool {
        match states.get(&node) {
            Some(VisitState::Visiting) => return true,
            Some(VisitState::Visited) => return false,
            None => {}
        }
        states.insert(node, VisitState::Visiting);
        if let Some(children) = adjacency.get(&node) {
            for (child, _) in children {
                if dfs(*child, adjacency, states) {
                    return true;
                }
            }
        states.insert(node, VisitState::Visited);
        false
    }
    for node in adjacency.keys() {
        if dfs(*node, &adjacency, &mut states) {
            return true;
    false
