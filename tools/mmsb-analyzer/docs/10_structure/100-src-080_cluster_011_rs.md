# Structure Group: src/080_cluster_011.rs

## File: src/080_cluster_011.rs

- Layer(s): 080_cluster_011.rs
- Language coverage: Rust (6)
- Element types: Function (6)
- Total elements: 6

### Elements

- [Rust | Function] `build_directory_dag` (line 0, pub)
  - Signature: `pub fn build_directory_dag (dir : & PathBuf) -> Result < DiGraph < PathBuf , () > > { let files : Vec < PathBuf > = w...`
  - Calls: collect, map, filter, filter_map, into_iter, walkdir::WalkDir::new, ok, unwrap_or, map, and_then, extension, path, to_str, into_path, collect, cloned, iter, build_module_map, crate::cluster_010::build_dependency_map, build_file_dag, Ok
- [Rust | Function] `build_file_dag` (line 0, pub(crate))
  - Signature: `pub (crate) fn build_file_dag (files : & [PathBuf] , dep_map : & HashMap < PathBuf , Vec < PathBuf > > ,) -> (DiGraph...`
  - Calls: DiGraph::new, HashMap::new, add_node, clone, insert, clone, get, get, add_edge
- [Rust | Function] `build_file_dependency_graph` (line 0, pub)
  - Signature: `pub fn build_file_dependency_graph (files : & [PathBuf]) -> Result < DiGraph < PathBuf , () > > { let file_set : Hash...`
  - Calls: collect, cloned, iter, build_module_map, crate::cluster_010::build_dependency_map, build_file_dag, Ok
- [Rust | Function] `build_module_map` (line 0, pub)
  - Signature: `pub fn build_module_map (files : & [PathBuf]) -> HashMap < String , PathBuf > { let mut map = HashMap :: new () ; for...`
  - Calls: HashMap::new, and_then, file_stem, to_str, crate::cluster_010::normalize_module_name, insert, clone, clone, and_then, parent, file_name, to_str, insert, crate::cluster_010::normalize_module_name, clone
- [Rust | Function] `export_program_cfg_to_path` (line 0, pub)
  - Signature: `pub fn export_program_cfg_to_path (result : & crate :: types :: AnalysisResult , call_edges : & [(String , String)] ,...`
  - Calls: HashMap::new, Vec::new, insert, clone, clone, to_string, unwrap_or, last, split, to_string, unwrap_or, last, split, contains_key, contains_key, push, join, std::fs::create_dir_all, join, crate::cluster_001::export_complete_program_dot, as_ref, to_string_lossy, join, to_str, to_str, status, args, std::process::Command::new, Ok
- [Rust | Function] `resolve_path` (line 0, pub)
  - Signature: `pub fn resolve_path (candidate : & Path , file_set : & HashSet < PathBuf > , module_map : & HashMap < String , PathBu...`
  - Calls: contains, Some, to_path_buf, and_then, file_stem, to_str, crate::cluster_010::normalize_module_name, get, Some, clone

