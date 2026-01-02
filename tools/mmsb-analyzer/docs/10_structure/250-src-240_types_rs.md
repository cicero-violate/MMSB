# Structure Group: src/240_types.rs

## File: src/240_types.rs

- Layer(s): 240_types.rs
- Language coverage: Rust (23)
- Element types: Enum (5), Impl (1), Struct (17)
- Total elements: 23

### Elements

- [Rust | Struct] `AnalysisResult` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct AnalysisResult { pub elements : Vec < CodeElement > , pub modules : Vec < ModuleInfo > ...`
- [Rust | Struct] `CallAnalysis` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct CallAnalysis { pub intra_file_calls : usize , pub inter_file_calls : Vec < (Pat...`
- [Rust | Struct] `CallGraphNode` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct CallGraphNode { pub function_name : String , pub file...`
- [Rust | Struct] `CfgEdge` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct CfgEdge { pub from : usize , pub to : usize , pub condition : Option < bool > , }`
- [Rust | Struct] `CfgNode` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct CfgNode { pub id : usize , pub node_type : NodeType , pub label : String , pub ...`
- [Rust | Struct] `CodeElement` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct CodeElement { pub name : String , pub file_path : Str...`
- [Rust | Struct] `DirectoryAnalysis` (line 0, pub)
  - Signature: `# [allow (dead_code)] # [derive (Debug , Clone)] pub struct DirectoryAnalysis { pub path : PathBuf , pub layer : Stri...`
- [Rust | Enum] `ElementType` (line 0, pub)
- [Rust | Struct] `FileLayerViolation` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FileLayerViolation { pub from : PathBuf , pub to : PathBuf , pub from_layer : S...`
- [Rust | Struct] `FileOrderEntry` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FileOrderEntry { pub current_path : PathBuf , pub canonical_order : usize , pub...`
- [Rust | Struct] `FileOrderingResult` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FileOrderingResult { pub ordered_files : Vec < FileOrderEntry > , pub violation...`
- [Rust | Struct] `FunctionCfg` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FunctionCfg { pub function : String , pub file_path : String , pub entry_id : u...`
- [Rust | Struct] `FunctionCluster` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FunctionCluster { pub members : Vec < String > , pub cohesion : f64 , pub sugge...`
- [Rust | Struct] `FunctionPlacement` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct FunctionPlacement { pub name : String , pub signature : String , pub current_fi...`
- [Rust | Struct] `JuliaElement` (line 0, pub)
  - Signature: `# [derive (Debug , Deserialize)] pub struct JuliaElement { pub element_type : String , pub name : String , pub file_p...`
- [Rust | Enum] `Language` (line 0, pub)
- [Rust | Struct] `ModuleInfo` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct ModuleInfo { pub name : String , pub file_path : Stri...`
- [Rust | Enum] `NodeType` (line 0, pub)
- [Rust | Struct] `OrderViolation` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct OrderViolation { pub file : PathBuf , pub current_position : usize , pub requir...`
- [Rust | Enum] `PlacementStatus` (line 0, pub)
- [Rust | Struct] `ProgramCFG` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct ProgramCFG { pub functions : HashMap < String , FunctionCfg > , pub call_edges : Vec < ...`
- [Rust | Enum] `Visibility` (line 0, pub)
- [Rust | Impl] `impl AnalysisResult { pub fn new () -> Self { Self { elements : Vec :: new () , modules : Vec :: new () , call_graph : HashMap :: new () , type_hierarchy : HashMap :: new () , cfgs : Vec :: new () , invariants : InvariantAnalysisResult :: new () , constraints : Vec :: new () , } } pub fn add_element (& mut self , element : CodeElement) { self . elements . push (element) ; } pub fn add_cfg (& mut self , cfg : FunctionCfg) { self . cfgs . push (cfg) ; } pub fn merge (& mut self , other : AnalysisResult) { self . elements . extend (other . elements) ; self . modules . extend (other . modules) ; self . call_graph . extend (other . call_graph) ; for (key , mut values) in other . type_hierarchy { self . type_hierarchy . entry (key) . or_insert_with (Vec :: new) . append (& mut values) ; } self . cfgs . extend (other . cfgs) ; self . invariants . invariants . extend (other . invariants . invariants) ; self . invariants . violations . extend (other . invariants . violations) ; self . invariants . layer_assignments . extend (other . invariants . layer_assignments) ; self . constraints . extend (other . constraints) ; } } . self_ty` (line 0, priv)

