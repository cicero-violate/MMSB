# Functions N-S

## Layer: 000_dependency_analysis.rs

### Rust Functions

#### `naming_score_for_file`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `to_string_lossy`
  - `file_name`
  - `to_string_lossy`
  - `file_stem`
  - `len`
  - `len`
  - `any`
  - `chars`
  - `is_uppercase`
  - `all`
  - `chars`
  - `is_ascii_lowercase`
  - `is_ascii_digit`
  - `contains`
  - `as_str`
  - `as_ref`
  - `fs::read_to_string`
  - `HashMap::new`
  - `Regex::new`
  - `captures_iter`
  - `get`
  - `to_lowercase`
  - `as_str`
  - `or_insert`
  - `entry`
  - `collect`
  - `into_iter`
  - `sort_by`
  - `cmp`
  - `collect`
  - `map`
  - `take`
  - `into_iter`
  - `collect`
  - `filter`
  - `map`
  - `split`
  - `to_lowercase`
  - `is_empty`
  - `all`
  - `chars`
  - `is_ascii_digit`
  - `count`
  - `filter`
  - `iter`
  - `any`
  - `iter`
  - `Some`

#### `order_julia_files_by_dependency`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `HashMap::new`
  - `BTreeSet::new`
  - `BTreeMap::new`
  - `Vec::new`
  - `LayerResolver::build`
  - `crate::cluster_001::julia_entry_paths`
  - `crate::cluster_001::detect_layer`
  - `insert`
  - `clone`
  - `insert`
  - `clone`
  - `clone`
  - `with_context`
  - `collect_julia_dependencies`
  - `is_absolute`
  - `clone`
  - `unwrap_or`
  - `map`
  - `parent`
  - `join`
  - `clone`
  - `exists`
  - `crate::cluster_001::detect_layer`
  - `insert`
  - `clone`
  - `insert`
  - `or_default`
  - `entry`
  - `clone`
  - `clone`
  - `clone`
  - `clone`
  - `push`
  - `clone`
  - `clone`
  - `resolve_module`
  - `insert`
  - `clone`
  - `insert`
  - `or_default`
  - `entry`
  - `clone`
  - `clone`
  - `clone`
  - `clone`
  - `push`
  - `clone`
  - `clone`
  - `crate::cluster_008::build_result`

#### `order_rust_files_by_dependency`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `crate::cluster_010::build_module_root_map`
  - `rust_entry_paths`
  - `HashMap::new`
  - `BTreeSet::new`
  - `BTreeMap::new`
  - `Vec::new`
  - `detect_layer`
  - `insert`
  - `clone`
  - `insert`
  - `clone`
  - `clone`
  - `with_context`
  - `collect_rust_dependencies`
  - `get`
  - `insert`
  - `clone`
  - `insert`
  - `or_default`
  - `entry`
  - `clone`
  - `clone`
  - `clone`
  - `clone`
  - `push`
  - `clone`
  - `clone`
  - `crate::cluster_008::build_result`

#### `ordered_by_name`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `to_vec`
  - `sort`
  - `collect`
  - `filter_map`
  - `into_iter`
  - `copied`
  - `get`

#### `run_analysis`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `join`
  - `RustAnalyzer::new`
  - `to_string`
  - `to_string_lossy`
  - `AnalysisResult::new`
  - `gather_rust_files`
  - `context`
  - `crate::dependency::order_rust_files_by_dependency`
  - `context`
  - `crate::dependency::analyze_file_ordering`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `analyze_file`
  - `merge`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `gather_julia_files`
  - `context`
  - `crate::dependency::order_julia_files_by_dependency`
  - `exists`
  - `JuliaAnalyzer::new`
  - `to_path_buf`
  - `clone`
  - `join`
  - `analyze_file`
  - `merge`
  - `is_some`
  - `is_some`
  - `Some`
  - `context`
  - `crate::dead_code_policy::load_policy`
  - `to_path_buf`
  - `to_path_buf`
  - `context`
  - `crate::dead_code_cli::run_dead_code_pipeline`
  - `crate::dead_code_filter::filter_dead_code_elements`
  - `ControlFlowAnalyzer::new`
  - `build_call_graph`
  - `InvariantDetector::new`
  - `detect_all`
  - `InvariantDetector::new`
  - `generate_constraints`
  - `FunctionCohesionAnalyzer::new`
  - `analyze`
  - `detect_clusters`
  - `DirectoryAnalyzer::new`
  - `to_path_buf`
  - `analyze`
  - `ReportGenerator::new`
  - `to_string`
  - `to_string_lossy`
  - `context`
  - `generate_all`
  - `export_program_cfg_to_path`
  - `call_edges`
  - `context`
  - `invariant_reporter::generate_invariant_report`
  - `context`
  - `invariant_reporter::export_constraints_json`
  - `Ok`

#### `rust_entry_paths`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `crate::layer_utilities::resolve_source_root`
  - `collect`
  - `filter`
  - `map`
  - `iter`
  - `join`
  - `exists`

## Layer: 010_layer_utilities.rs

### Rust Functions

#### `node_style`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public

#### `parse_cluster_members`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `filter_map`
  - `iter`
  - `rsplit_once`
  - `Some`
  - `PathBuf::from`
  - `to_string`

#### `sort_structural_items`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `len`
  - `HashMap::new`
  - `enumerate`
  - `iter`
  - `push`
  - `or_default`
  - `entry`
  - `clone`
  - `structural_layer_value`
  - `structural_layer_value`
  - `Some`
  - `Some`
  - `Some`
  - `Some`
  - `push`
  - `enumerate`
  - `iter`
  - `get`
  - `push`
  - `Vec::with_capacity`
  - `collect`
  - `filter`
  - `is_empty`
  - `sort_by`
  - `structural_cmp`
  - `remove`
  - `push`
  - `saturating_sub`
  - `push`
  - `len`
  - `sort_by`
  - `Vec::with_capacity`
  - `push`
  - `clone`

#### `structural_cmp`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `structural_layer_value`
  - `structural_layer_value`
  - `structural_layer_value`
  - `structural_layer_value`
  - `saturating_mul`
  - `saturating_mul`
  - `then_with`
  - `then_with`
  - `then_with`
  - `then_with`
  - `then_with`
  - `cmp`
  - `cmp`
  - `cmp`
  - `cmp`
  - `cmp`
  - `cmp`

#### `structural_layer_value`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Crate
- **Calls:**
  - `unwrap_or`
  - `and_then`
  - `as_ref`
  - `layer_prefix_value`

## Layer: 060_module_resolution.rs

### Rust Functions

#### `normalize_module_name`

- **File:** src/060_module_resolution.rs:0
- **Visibility:** Public
- **Calls:**
  - `find`
  - `all`
  - `chars`
  - `is_ascii_digit`
  - `to_string`
  - `to_string`

#### `resolve_module`

- **File:** src/060_module_resolution.rs:0
- **Visibility:** Public
- **Calls:**
  - `normalize_module_name`
  - `get`
  - `Some`
  - `clone`
  - `or_else`
  - `or_else`
  - `map`
  - `find`
  - `iter`
  - `clone`
  - `map`
  - `find`
  - `iter`
  - `starts_with`
  - `as_str`
  - `clone`
  - `crate::cluster_011::resolve_path`
  - `PathBuf::from`

#### `resolve_module_name`

- **File:** src/060_module_resolution.rs:0
- **Visibility:** Private
- **Calls:**
  - `unwrap_or`
  - `next`
  - `split`
  - `resolve_module`

## Layer: 080_cluster_011.rs

### Rust Functions

#### `resolve_path`

- **File:** src/080_cluster_011.rs:0
- **Visibility:** Public
- **Calls:**
  - `contains`
  - `Some`
  - `to_path_buf`
  - `and_then`
  - `file_stem`
  - `to_str`
  - `crate::cluster_010::normalize_module_name`
  - `get`
  - `Some`
  - `clone`

## Layer: 120_cluster_006.rs

### Rust Functions

#### `order_directories`

- **File:** src/120_cluster_006.rs:0
- **Visibility:** Public
- **Calls:**
  - `common_root`
  - `HashSet::new`
  - `map`
  - `parent`
  - `starts_with`
  - `insert`
  - `clone`
  - `map`
  - `parent`
  - `collect`
  - `into_iter`
  - `sort_by`
  - `crate::cluster_008::compare_path_components`
  - `HashMap::new`
  - `enumerate`
  - `iter`
  - `insert`
  - `clone`
  - `map`
  - `parent`
  - `get`
  - `map`
  - `parent`
  - `get`
  - `insert`
  - `collect`
  - `filter_map`
  - `enumerate`
  - `iter`
  - `Some`
  - `Vec::with_capacity`
  - `len`
  - `next`
  - `iter`
  - `remove`
  - `push`
  - `clone`
  - `clone`
  - `insert`
  - `len`
  - `len`
  - `enumerate`
  - `iter`
  - `push`
  - `clone`

#### `strip_numeric_prefix`

- **File:** src/120_cluster_006.rs:0
- **Visibility:** Crate
- **Calls:**
  - `Lazy::new`
  - `unwrap`
  - `Regex::new`
  - `unwrap_or`
  - `map`
  - `and_then`
  - `captures`
  - `get`
  - `as_str`

## Layer: 170_layer_utilities.rs

### Rust Functions

#### `resolve_source_root`

- **File:** src/170_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `join`
  - `exists`
  - `is_dir`
  - `to_path_buf`

#### `run_dead_code_pipeline`

- **File:** src/170_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `gather_rust_files`
  - `HashMap::new`
  - `TestBoundaries::default`
  - `detect_intent_signals`
  - `as_ref`
  - `merge_intent_map`
  - `detect_test_modules`
  - `extend`
  - `detect_test_symbols`
  - `extend`
  - `is_test_path`
  - `insert`
  - `clone`
  - `build_call_graph`
  - `collect_entrypoints`
  - `as_ref`
  - `collect_exports`
  - `Vec::new`
  - `classify_symbol`
  - `as_ref`
  - `contains_key`
  - `contains`
  - `is_reachable`
  - `clone`
  - `PathBuf::from`
  - `reason_for_category`
  - `assign_confidence`
  - `is_public_api`
  - `recommend_action`
  - `push`
  - `to_string`
  - `to_string`
  - `display`
  - `len`
  - `build_report`
  - `to_rfc3339`
  - `chrono::Local::now`
  - `write_outputs`
  - `Ok`

## Layer: 211_dead_code_doc_comment_scanner.rs

### Rust Functions

#### `scan_doc_comments`

- **File:** src/211_dead_code_doc_comment_scanner.rs:0
- **Visibility:** Public
- **Calls:**
  - `unwrap_or_default`
  - `std::fs::read_to_string`
  - `syn::parse_file`
  - `HashMap::new`
  - `HashMap::new`
  - `item_name`
  - `extract_doc_markers`
  - `item_attrs`
  - `is_empty`
  - `extend`
  - `or_default`
  - `entry`

## Layer: 230_dead_code_attribute_parser.rs

### Rust Functions

#### `parse_mmsb_latent_attr`

- **File:** src/230_dead_code_attribute_parser.rs:0
- **Visibility:** Public
- **Calls:**
  - `unwrap_or_default`
  - `std::fs::read_to_string`
  - `syn::parse_file`
  - `HashMap::new`
  - `HashMap::new`
  - `item_name`
  - `collect_latent_attrs`
  - `item_attrs`
  - `is_empty`
  - `extend`
  - `or_default`
  - `entry`

#### `scan_file_attributes`

- **File:** src/230_dead_code_attribute_parser.rs:0
- **Visibility:** Public
- **Calls:**
  - `unwrap_or_default`
  - `std::fs::read_to_string`
  - `syn::parse_file`
  - `Vec::new`
  - `Vec::new`
  - `item_name`
  - `collect_latent_attrs`
  - `item_attrs`
  - `push`
  - `clone`
  - `to_path_buf`
  - `clone`

#### `scan_intent_tags`

- **File:** src/230_dead_code_attribute_parser.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `parse_mmsb_latent_attr`
  - `push`
  - `clone`
  - `to_path_buf`
  - `clone`
  - `scan_doc_comments`
  - `push`
  - `clone`
  - `to_path_buf`
  - `check_planned_directory`
  - `collect_symbols`
  - `push`
  - `to_path_buf`

## Layer: 280_file_ordering.rs

### Rust Functions

#### `parallel_build_file_dag`

- **File:** src/280_file_ordering.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `map`
  - `par_iter`
  - `crate::dependency::build_directory_dag`
  - `DiGraph::new`
  - `HashMap::new`
  - `node_indices`
  - `clone`
  - `or_insert_with`
  - `entry`
  - `clone`
  - `add_node`
  - `edge_indices`
  - `edge_endpoints`
  - `clone`
  - `clone`
  - `expect`
  - `get`
  - `expect`
  - `get`
  - `add_edge`
  - `Ok`

## Layer: 330_markdown_report.rs

### Rust Functions

#### `path_common_prefix_len`

- **File:** src/330_markdown_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `zip`
  - `components`
  - `components`

#### `resolve_required_layer_path`

- **File:** src/330_markdown_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `collect_directory_files`
  - `collect`
  - `filter`
  - `into_iter`
  - `unwrap_or`
  - `map`
  - `and_then`
  - `file_name`
  - `to_str`
  - `is_empty`
  - `join`
  - `unwrap_or`
  - `parent`
  - `unwrap_or`
  - `parent`
  - `unwrap_or`
  - `parent`
  - `path_common_prefix_len`
  - `count`
  - `components`
  - `Some`
  - `unwrap_or_else`
  - `join`
  - `unwrap_or`
  - `parent`

## Layer: 350_agent_cli.rs

### Rust Functions

#### `query_function`

- **File:** src/350_agent_cli.rs:0
- **Visibility:** Private
- **Calls:**
  - `load_invariants`
  - `AgentConscience::new`
  - `query_allowed_actions`
  - `serde_json::to_string_pretty`
  - `Ok`

#### `run_agent_cli`

- **File:** src/350_agent_cli.rs:0
- **Visibility:** Public
- **Calls:**
  - `AgentCli::parse`
  - `check_action`
  - `query_function`
  - `list_invariants`
  - `show_stats`
  - `Ok`

#### `show_stats`

- **File:** src/350_agent_cli.rs:0
- **Visibility:** Private
- **Calls:**
  - `load_invariants`
  - `AgentConscience::new`
  - `stats`
  - `Ok`

## Layer: 400_dead_code_intent.rs

### Rust Functions

#### `planned_directory_intent`

- **File:** src/400_dead_code_intent.rs:0
- **Visibility:** Crate
- **Calls:**
  - `check_planned_directory`
  - `IntentMap::new`
  - `HashMap::new`
  - `collect_symbols`
  - `push`
  - `or_default`
  - `entry`

## Layer: 450_dead_code_actions.rs

### Rust Functions

#### `recommend_action`

- **File:** src/450_dead_code_actions.rs:0
- **Visibility:** Public

## Layer: 480_dead_code_filter.rs

### Rust Functions

#### `should_exclude_from_analysis`

- **File:** src/480_dead_code_filter.rs:0
- **Visibility:** Public

## Layer: 500_dead_code_cli.rs

### Rust Functions

#### `reason_for_category`

- **File:** src/500_dead_code_cli.rs:0
- **Visibility:** Crate
- **Calls:**
  - `to_string`
  - `to_string`
  - `to_string`
  - `to_string`
  - `to_string`
  - `to_string`

## Layer: 520_dead_code_policy.rs

### Rust Functions

#### `parse_bool`

- **File:** src/520_dead_code_policy.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_str`
  - `to_ascii_lowercase`
  - `trim`
  - `Some`
  - `Some`

#### `parse_list`

- **File:** src/520_dead_code_policy.rs:0
- **Visibility:** Private
- **Calls:**
  - `to_string`
  - `trim`
  - `strip_prefix`
  - `to_string`
  - `strip_suffix`
  - `to_string`
  - `collect`
  - `filter`
  - `map`
  - `split`
  - `to_string`
  - `trim_matches`
  - `trim_matches`
  - `trim`
  - `is_empty`

#### `parse_policy`

- **File:** src/520_dead_code_policy.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `lines`
  - `trim`
  - `is_empty`
  - `starts_with`
  - `starts_with`
  - `split_once`
  - `trim`
  - `trim`
  - `collect`
  - `map`
  - `into_iter`
  - `parse_list`
  - `join`
  - `collect`
  - `map`
  - `into_iter`
  - `parse_list`
  - `join`
  - `parse_list`
  - `unwrap_or`
  - `parse_bool`

## Layer: 540_dead_code_report_split.rs

### Rust Functions

#### `plan_options`

- **File:** src/540_dead_code_report_split.rs:0
- **Visibility:** Private
- **Calls:**
  - `push_str`

## Layer: 620_correction_plan_serializer.rs

### Rust Functions

#### `serialize_correction_plan`

- **File:** src/620_correction_plan_serializer.rs:0
- **Visibility:** Public

#### `serialize_correction_plans`

- **File:** src/620_correction_plan_serializer.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `map`
  - `zip`
  - `zip`
  - `iter`
  - `iter`
  - `iter`
  - `serialize_correction_plan`

## Layer: 640_correction_intelligence_report.rs

### Rust Functions

#### `parse_phase2_cluster_plan`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `fs::read_to_string`
  - `map_err`
  - `Regex::new`
  - `std::io::Error::new`
  - `to_string`
  - `map_err`
  - `Regex::new`
  - `std::io::Error::new`
  - `to_string`
  - `Vec::new`
  - `lines`
  - `captures`
  - `take`
  - `push`
  - `Some`
  - `PathBuf::from`
  - `Vec::new`
  - `captures`
  - `as_mut`
  - `push`
  - `to_string`
  - `PathBuf::from`
  - `push`
  - `Ok`

#### `plan_verification_scope`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `affected_files`
  - `action_module`
  - `push`
  - `clone`
  - `estimate_verification_time`

#### `predict_violations`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `find_callers`
  - `is_empty`
  - `push`
  - `is_empty`
  - `push`
  - `move_violates_invariant`
  - `push`
  - `symbol_exists`
  - `push`
  - `find_reference_files`
  - `is_empty`
  - `push`
  - `push`
  - `push`
  - `push`

#### `simulate_action`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Crate
- **Calls:**
  - `clone`

#### `symbol_exists`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Crate
- **Calls:**
  - `any`
  - `iter`

## Layer: root

### Rust Functions

#### `project_conflict_reason`

- **File:** src/admission_composition_artifact.rs:0
- **Visibility:** Private
- **Calls:**
  - `to_string`
  - `display`
  - `clone`
  - `clone`
  - `to_string`
  - `clone`

#### `project_invariants_touched`

- **File:** src/admission_composition_artifact.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `is_empty`
  - `push`
  - `to_string`
  - `sort`

#### `project_state`

- **File:** src/admission_composition_artifact.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `len`
  - `project_invariants_touched`

#### `read_artifact`

- **File:** src/admission_composition_artifact.rs:0
- **Visibility:** Public
- **Calls:**
  - `std::fs::read_to_string`
  - `serde_json::from_str`
  - `Ok`

