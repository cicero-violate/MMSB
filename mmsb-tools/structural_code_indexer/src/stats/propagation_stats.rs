//! Propagation Statistics Computation
//!
//! **Purely observational** - does not modify the graph.

use mmsb_core::adaptive::PropagationStats;
use mmsb_core::dag::DependencyGraph;
use mmsb_core::types::PageID;
use std::collections::HashMap;

/// Compute propagation statistics from dependency graph
///
/// **Read-only:** Does not modify graph
pub fn compute_propagation_stats(dag: &DependencyGraph) -> PropagationStats {
    let mut stats = PropagationStats::new();
    
    // Compute fanout for each page
    let mut fanout: HashMap<PageID, usize> = HashMap::new();
    let mut fanin: HashMap<PageID, usize> = HashMap::new();
    
    for (from, to, _) in dag.edges() {
        *fanout.entry(from).or_insert(0) += 1;
        *fanin.entry(to).or_insert(0) += 1;
    }
    
    // Populate stats with fanout counts
    for (page, count) in fanout {
        stats.fanout_per_page.insert(page, count);
    }
    
    // Estimate derived delta counts from fanout
    // (In real usage, this would come from actual propagation runs)
    for (page, count) in stats.fanout_per_page.iter() {
        stats.derived_delta_count.insert(*page, *count);
    }
    
    stats.total_propagations = stats.fanout_per_page.values().sum();
    
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmsb_core::dag::{build_dependency_graph, EdgeType, StructuralOp};
    use mmsb_core::types::PageID;
    
    #[test]
    fn test_compute_stats() {
        let page_a = PageID(1);
        let page_b = PageID(2);
        let page_c = PageID(3);
        
        // A → B, A → C
        let dag = build_dependency_graph(&[
            StructuralOp::AddEdge {
                from: page_a,
                to: page_b,
                edge_type: EdgeType::Data,
            },
            StructuralOp::AddEdge {
                from: page_a,
                to: page_c,
                edge_type: EdgeType::Data,
            },
        ]);
        
        let stats = compute_propagation_stats(&dag);
        
        assert_eq!(stats.fanout_per_page.get(&page_a), Some(&2));
        assert_eq!(stats.total_propagations, 2);
    }
}
