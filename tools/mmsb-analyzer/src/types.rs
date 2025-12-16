//! Type definitions for code analysis

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Language {
    Rust,
    Julia,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElementType {
    Struct,
    Enum,
    Trait,
    Impl,
    Function,
    Module,
    Const,
    Static,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeElement {
    pub element_type: ElementType,
    pub name: String,
    pub file_path: String,
    pub line_number: usize,
    pub language: Language,
    pub layer: String,
    pub signature: String,
    pub calls: Vec<String>,
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

#[derive(Debug)]
pub struct AnalysisResult {
    pub elements: Vec<CodeElement>,
    pub modules: Vec<ModuleInfo>,
    pub call_graph: HashMap<String, CallGraphNode>,
    pub type_hierarchy: HashMap<String, Vec<String>>,
}

impl AnalysisResult {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            modules: Vec::new(),
            call_graph: HashMap::new(),
            type_hierarchy: HashMap::new(),
        }
    }
    
    pub fn add_element(&mut self, element: CodeElement) {
        self.elements.push(element);
    }
    
    pub fn add_module(&mut self, module: ModuleInfo) {
        self.modules.push(module);
    }
    
    pub fn merge(&mut self, other: AnalysisResult) {
        self.elements.extend(other.elements);
        self.modules.extend(other.modules);
        self.call_graph.extend(other.call_graph);
        for (key, mut values) in other.type_hierarchy {
            self.type_hierarchy.entry(key).or_insert_with(Vec::new).append(&mut values);
        }
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
