//! Control flow and call graph analysis

use crate::types::*;
use crate::utilities::compress_path;
use petgraph::dot::Dot;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

pub struct ControlFlowAnalyzer {
    graph: DiGraph<String, String>,
    node_map: HashMap<String, NodeIndex>,
    cfgs: Vec<FunctionCfg>,
}

impl ControlFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            cfgs: Vec::new(),
        }
    }

    pub fn build_call_graph(&mut self, result: &AnalysisResult) {
        self.graph = DiGraph::new();
        self.node_map.clear();
        self.cfgs = result.cfgs.clone();

        // First pass: create nodes for all functions
        for elem in &result.elements {
            if matches!(elem.element_type, ElementType::Function) {
                let node_name = format!("{}::{}", compress_path(&elem.file_path), elem.name);
                let node_idx = self.graph.add_node(node_name.clone());
                self.node_map.insert(node_name, node_idx);
            }
        }

        let total_funcs = result
            .elements
            .iter()
            .filter(|elem| matches!(elem.element_type, ElementType::Function))
            .count();
        let mut processed = 0usize;
        let mut last_report = 0usize;

        // Second pass: create edges for function calls
        for elem in &result.elements {
            if matches!(elem.element_type, ElementType::Function) {
                processed += 1;
                if processed % 50 == 0 || processed == total_funcs {
                    if processed != last_report {
                        println!(
                            "  Call graph progress: {}/{} functions",
                            processed, total_funcs
                        );
                        last_report = processed;
                    }
                }
                let caller_name = format!("{}::{}", compress_path(&elem.file_path), elem.name);

                if let Some(&caller_idx) = self.node_map.get(&caller_name) {
                    for called in &elem.calls {
                        // Try to find the called function
                        for target_elem in &result.elements {
                            if matches!(target_elem.element_type, ElementType::Function)
                                && (target_elem.name == *called
                                    || called.ends_with(&target_elem.name))
                            {
                                let callee_name = format!(
                                    "{}::{}",
                                    compress_path(&target_elem.file_path),
                                    target_elem.name
                                );
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

    #[allow(dead_code)]
    pub fn generate_dot(&self) -> String {
        format!("{:?}", Dot::new(&self.graph))
    }

    pub fn generate_mermaid(&self) -> String {
        let mut output = String::from("```mermaid\ngraph TD\n");

        for node_idx in self.graph.node_indices() {
            let node_name = &self.graph[node_idx];
            let safe_name = sanitize_identifier(node_name);
            output.push_str(&format!("    {}[\"{}\"]\n", safe_name, node_name));
        }

        for edge in self.graph.edge_indices() {
            if let Some((source, target)) = self.graph.edge_endpoints(edge) {
                let source_name = &self.graph[source];
                let target_name = &self.graph[target];
                let safe_source = sanitize_identifier(source_name);
                let safe_target = sanitize_identifier(target_name);
                output.push_str(&format!("    {} --> {}\n", safe_source, safe_target));
            }
        }

        output.push_str("```\n");
        output
    }

    pub fn call_edges(&self) -> Vec<(String, String)> {
        let mut edges = Vec::new();
        for edge_idx in self.graph.edge_indices() {
            if let Some((source, target)) = self.graph.edge_endpoints(edge_idx) {
                edges.push((self.graph[source].clone(), self.graph[target].clone()));
            }
        }
        edges
    }

    pub fn cfgs(&self) -> &[FunctionCfg] {
        &self.cfgs
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

    fn dfs_depth(
        &self,
        node: NodeIndex,
        visited: &mut std::collections::HashSet<NodeIndex>,
    ) -> usize {
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

fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

