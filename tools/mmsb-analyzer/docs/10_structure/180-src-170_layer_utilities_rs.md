# Structure Group: src/170_layer_utilities.rs

## File: src/170_layer_utilities.rs

- Layer(s): 170_layer_utilities.rs
- Language coverage: Rust (5)
- Element types: Function (4), Struct (1)
- Total elements: 5

### Elements

- [Rust | Struct] `Args` (line 0, priv)
  - Signature: `# [derive (Parser , Debug)] # [command (name = "mmsb-analyzer")] # [command (about = "MMSB Intelligence Substrate Ana...`
- [Rust | Function] `allow_analysis_dir` (line 0, pub)
  - Signature: `# [doc = " Checks if a directory should be included in analysis"] pub fn allow_analysis_dir (root : & Path , dir : & ...`
  - Calls: unwrap_or, and_then, file_name, to_str, starts_with, strip_prefix, any, components, unwrap_or, to_str, as_os_str, starts_with
- [Rust | Function] `main` (line 0, pub)
  - Signature: `pub fn main () -> Result < () > { let args = Args :: parse () ; let root_path = std :: env :: current_dir () ? . join...`
  - Calls: Args::parse, canonicalize, join, std::env::current_dir, unwrap_or_else, canonicalize, join, std::env::current_dir, join, unwrap, std::env::current_dir, ok, std::fs::create_dir_all, unwrap_or, canonicalize, run_analysis
- [Rust | Function] `resolve_source_root` (line 0, pub)
  - Signature: `# [doc = " Resolves the source root directory from a given root path"] pub fn resolve_source_root (root : & Path) -> ...`
  - Calls: join, exists, is_dir, to_path_buf
- [Rust | Function] `run_dead_code_pipeline` (line 0, pub)
  - Signature: `pub fn run_dead_code_pipeline (elements : & [CodeElement] , config : & DeadCodeRunConfig ,) -> Result < DeadCodeRepor...`
  - Calls: gather_rust_files, HashMap::new, TestBoundaries::default, detect_intent_signals, as_ref, merge_intent_map, detect_test_modules, extend, detect_test_symbols, extend, is_test_path, insert, clone, build_call_graph, collect_entrypoints, as_ref, collect_exports, Vec::new, classify_symbol, as_ref, contains_key, contains, is_reachable, clone, PathBuf::from, reason_for_category, assign_confidence, is_public_api, recommend_action, push, to_string, to_string, display, len, build_report, to_rfc3339, chrono::Local::now, write_outputs, Ok

