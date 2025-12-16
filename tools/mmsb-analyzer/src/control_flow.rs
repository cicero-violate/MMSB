//! Control flow and call graph analysis

use crate::types::*;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::dot::Dot;
use std::collections::HashMap;

pub struct ControlFlowAnalyzer {
    graph: DiGraph<String, String>,
    node_map: HashMap<String, NodeIndex>,
}

impl ControlFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }
    
    pub fn build_call_graph(&mut self, result: &AnalysisResult) {
        // First pass: create nodes for all functions
        for elem in &result.elements {
            if matches!(elem.element_type, ElementType::Function) {
                let node_name = format!("{}::{}", elem.file_path, elem.name);
                let node_idx = self.graph.add_node(node_name.clone());
                self.node_map.insert(node_name, node_idx);
            }
        }
        
        // Second pass: create edges for function calls
        for elem in &result.elements {
            if matches!(elem.element_type, ElementType::Function) {
                let caller_name = format!("{}::{}", elem.file_path, elem.name);
                
                if let Some(&caller_idx) = self.node_map.get(&caller_name) {
                    for called in &elem.calls {
                        // Try to find the called function
                        for target_elem in &result.elements {
                            if matches!(target_elem.element_type, ElementType::Function) 
                                && (target_elem.name == *called || called.ends_with(&target_elem.name)) {
                                let callee_name = format!("{}::{}", target_elem.file_path, target_elem.name);
                                if let Some(&callee_idx) = self.node_map.get(&callee_name) {
                                    self.graph.add_edge(caller_idx, callee_idx, called.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    pub fn generate_dot(&self) -> String {
        format!("{:?}", Dot::new(&self.graph))
    }
    
    pub fn generate_mermaid(&self) -> String {
        let mut output = String::from("```mermaid\ngraph TD\n");
        
        for node_idx in self.graph.node_indices() {
            let node_name = &self.graph[node_idx];
            let safe_name = node_name.replace("::", "_").replace("/", "_").replace(".", "_");
            output.push_str(&format!("    {}[\"{}\"]\n", safe_name, node_name));
        }
        
        for edge in self.graph.edge_indices() {
            if let Some((source, target)) = self.graph.edge_endpoints(edge) {
                let source_name = &self.graph[source];
                let target_name = &self.graph[target];
                let safe_source = source_name.replace("::", "_").replace("/", "_").replace(".", "_");
                let safe_target = target_name.replace("::", "_").replace("/", "_").replace(".", "_");
                output.push_str(&format!("    {} --> {}\n", safe_source, safe_target));
            }
        }
        
        output.push_str("```\n");
        output
    }
    
    pub fn get_statistics(&self) -> CallGraphStats {
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();
        
        let mut max_depth = 0;
        let mut leaf_functions = 0;
        
        for node_idx in self.graph.node_indices() {
            let outgoing = self.graph.edges(node_idx).count();
            if outgoing == 0 {
                leaf_functions += 1;
            }
            
            // Simple depth calculation (could be improved with proper traversal)
            let depth = self.calculate_depth(node_idx);
            if depth > max_depth {
                max_depth = depth;
            }
        }
        
        CallGraphStats {
            total_functions: node_count,
            total_calls: edge_count,
            max_depth,
            leaf_functions,
        }
    }
    
    fn calculate_depth(&self, start: NodeIndex) -> usize {
        let mut visited = std::collections::HashSet::new();
        self.dfs_depth(start, &mut visited)
    }
    
    fn dfs_depth(&self, node: NodeIndex, visited: &mut std::collections::HashSet<NodeIndex>) -> usize {
        if visited.contains(&node) {
            return 0;
        }
        visited.insert(node);
        
        let mut max = 0;
        for neighbor in self.graph.neighbors(node) {
            let depth = self.dfs_depth(neighbor, visited);
            if depth > max {
                max = depth;
            }
        }
        
        visited.remove(&node);
        max + 1
    }
}

pub struct CallGraphStats {
    pub total_functions: usize,
    pub total_calls: usize,
    pub max_depth: usize,
    pub leaf_functions: usize,
}
