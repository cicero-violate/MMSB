use mmsb_memory::dag::dependency_graph::DependencyGraph;
use mmsb_primitives::PageID;

let mut dag = DependencyGraph::new();

dag.add_edge(PageID(1), PageID(2), EdgeType::Materializes);
dag.add_edge(PageID(2), PageID(3), EdgeType::DependsOn);
