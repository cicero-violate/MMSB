use crate::dag::DependencyGraph;
use crate::page::{Delta, PageID};
use std::collections::HashSet;

pub fn compute_affected_pages(
    dag: &DependencyGraph,
    changed_pages: &[PageID],
) -> HashSet<PageID> {
    let mut affected = HashSet::new();
    
    for page_id in changed_pages {
        let descendants = dag.descendants(*page_id);
        affected.extend(descendants);
    }
    
    affected
}

pub fn propagate_delta_to_descendants(
    dag: &DependencyGraph,
    source_delta: &Delta,
) -> Vec<PageID> {
    let descendants = dag.descendants(source_delta.page_id);
    descendants.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{StructuralOp, EdgeType};
    use crate::page::{Delta, DeltaID, Epoch, Source};

    #[test]
    fn compute_affected_includes_descendants() {
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
        let dag = crate::dag::build_dependency_graph(&ops);

        let affected = compute_affected_pages(&dag, &[PageID(1)]);
        assert!(affected.contains(&PageID(1)));
        assert!(affected.contains(&PageID(2)));
        assert!(affected.contains(&PageID(3)));
    }
}
