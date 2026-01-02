//! Type definitions for code analysis
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// Import from sibling modules
use crate::invariant_types::InvariantAnalysisResult;
use crate::refactor_constraints::RefactorConstraint;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ElementType {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Language {
    Rust,
    Julia,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeElement {
    pub name: String,
    pub file_path: String,
    pub line_number: usize,
    pub element_type: ElementType,
    pub signature: String,
    pub visibility: Visibility,
    pub generic_params: Vec<String>,
    pub language: Language,
    pub layer: String,
    pub calls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub invariants: InvariantAnalysisResult,
    pub constraints: Vec<RefactorConstraint>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DirectoryAnalysis {
    pub path: PathBuf,
    pub layer: String,
    pub parent: Option<PathBuf>,
    pub files: Vec<PathBuf>,
    pub subdirectories: Vec<DirectoryAnalysis>,
    pub has_sources: bool,
}

#[derive(Debug, Clone)]
pub struct FileOrderEntry {
    pub current_path: PathBuf,
    pub canonical_order: usize,
    pub suggested_name: String,
    pub needs_rename: bool,
}

#[derive(Debug, Clone)]
pub struct OrderViolation {
    pub file: PathBuf,
    pub current_position: usize,
    pub required_position: usize,
    pub blocking_dependencies: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct FileLayerViolation {
    pub from: PathBuf,
    pub to: PathBuf,
    pub from_layer: String,
    pub to_layer: String,
}

#[derive(Debug, Clone)]
pub struct FileOrderingResult {
    pub ordered_files: Vec<FileOrderEntry>,
    pub violations: Vec<OrderViolation>,
    pub layer_violations: Vec<FileLayerViolation>,
    pub ordered_directories: Vec<PathBuf>,
    pub cycles: Vec<Vec<PathBuf>>,
}

#[derive(Debug, Clone)]
pub struct CallAnalysis {
    pub intra_file_calls: usize,
    pub inter_file_calls: Vec<(PathBuf, usize)>,
    pub calls_from_same_file: usize,
    pub calls_from_other_files: Vec<(PathBuf, usize)>,
    pub same_file_type_refs: usize,
    pub other_file_type_refs: usize,
}

#[derive(Debug, Clone)]
pub enum PlacementStatus {
    Correct,
    ShouldMove { reason: String, impact: f64 },
    Orphaned { suggested_module: String },
    LayerViolation { current_layer: String, required_layer: String },
}

#[derive(Debug, Clone)]
pub struct FunctionPlacement {
    pub name: String,
    pub signature: String,
    pub current_file: PathBuf,
    pub suggested_file: Option<PathBuf>,
    pub placement_status: PlacementStatus,
    pub cohesion_score: f64,
    pub call_analysis: CallAnalysis,
}

#[derive(Debug, Clone)]
pub struct FunctionCluster {
    pub members: Vec<String>,
    pub cohesion: f64,
    pub suggested_file: Option<PathBuf>,
}

impl AnalysisResult {
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
            modules: Vec::new(),
            call_graph: HashMap::new(),
            type_hierarchy: HashMap::new(),
            cfgs: Vec::new(),
            invariants: InvariantAnalysisResult::new(),
            constraints: Vec::new(),
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

        // Merge invariants
        self.invariants.invariants.extend(other.invariants.invariants);
        self.invariants.violations.extend(other.invariants.violations);
        self.invariants.layer_assignments.extend(other.invariants.layer_assignments);

        // Merge constraints
        self.constraints.extend(other.constraints);
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
