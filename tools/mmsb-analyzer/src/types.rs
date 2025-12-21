//! Type definitions for code analysis
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeElement {
    pub name: String,
    pub file_path: String,
    pub line_number: usize,
    pub element_type: String,
    pub signature: String,
    pub visibility: Visibility,
    pub generic_params: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Crate,
    Private,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub file_path: String,
    pub imports: Vec<String>,
    pub exports: Vec<String>,
    pub submodules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallGraphNode {
    pub function_name: String,
    pub file_path: String,
    pub calls: Vec<String>,
    pub called_by: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeType {
    Entry,
    Exit,
    BasicBlock,
    Branch,
    LoopHeader,
}

#[derive(Debug, Clone)]
pub struct CfgNode {
    pub id: usize,
    pub node_type: NodeType,
    pub label: String,
    pub lines: Vec<u32>,  // Source line numbers (empty for Rust currently)
}

#[derive(Debug, Clone)]
pub struct CfgEdge {
    pub from: usize,
    pub to: usize,
    pub condition: Option<bool>,  // Some(true)=taken/true branch, Some(false)=false/else, None=unconditional
}

#[derive(Debug, Clone)]
pub struct FunctionCfg {
    pub function: String,
    pub file_path: String,
    pub entry_id: usize,
    pub exit_id: usize,
    pub nodes: Vec<CfgNode>,
    pub edges: Vec<CfgEdge>,
    pub branch_count: usize,
    pub loop_count: usize,
}

#[derive(Debug)]
pub struct ProgramCFG {
    pub functions: HashMap<String, FunctionCfg>,  // Key: function name (assume unique)
    pub call_edges: Vec<(String, String)>,  // (caller, callee)
}

#[derive(Debug)]
pub struct AnalysisResult {
    pub elements: Vec<CodeElement>,
    pub modules: Vec<ModuleInfo>,
    pub call_graph: HashMap<String, CallGraphNode>,
    pub type_hierarchy: HashMap<String, Vec<String>>,
    pub cfgs: Vec<FunctionCfg>,
}

impl AnalysisResult {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            modules: Vec::new(),
            call_graph: HashMap::new(),
            type_hierarchy: HashMap::new(),
            cfgs: Vec::new(),
        }
    }

    pub fn add_element(&mut self, element: CodeElement) {
        self.elements.push(element);
    }

    pub fn add_cfg(&mut self, cfg: FunctionCfg) {
        self.cfgs.push(cfg);
    }

    pub fn merge(&mut self, other: AnalysisResult) {
        self.elements.extend(other.elements);
        self.modules.extend(other.modules);
        self.call_graph.extend(other.call_graph);
        for (key, mut values) in other.type_hierarchy {
            self.type_hierarchy
                .entry(key)
                .or_insert_with(Vec::new)
                .append(&mut values);
        }
        self.cfgs.extend(other.cfgs);
    }
}

#[derive(Debug, Deserialize)]
pub struct JuliaElement {
    pub element_type: String,
    pub name: String,
    pub file_path: String,
    pub line_number: usize,
    pub signature: String,
    pub calls: Vec<String>,
}
