# Functions G-M

## Layer: 000_dependency_analysis.rs

### Rust Functions

#### `gather_julia_files`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `crate::layer_utilities::resolve_source_root`
  - `collect`
  - `map`
  - `filter`
  - `filter`
  - `filter_map`
  - `filter_entry`
  - `into_iter`
  - `WalkDir::new`
  - `depth`
  - `is_dir`
  - `file_type`
  - `crate::layer_utilities::allow_analysis_dir`
  - `path`
  - `ok`
  - `map_or`
  - `extension`
  - `path`
  - `unwrap_or`
  - `strip_prefix`
  - `path`
  - `path`
  - `count`
  - `components`
  - `starts_with`
  - `path`
  - `join`
  - `into_path`

#### `julia_entry_paths`

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

#### `layer_constrained_sort`

- **File:** src/000_dependency_analysis.rs:0
- **Visibility:** Public
- **Calls:**
  - `BTreeMap::new`
  - `node_indices`
  - `unwrap_or_else`
  - `cloned`
  - `get`
  - `to_string`
  - `unwrap_or`
  - `layer_prefix_value`
  - `push`
  - `or_default`
  - `entry`
  - `Vec::new`
  - `topo_sort_within`
  - `extend`
  - `Ok`

## Layer: 010_layer_utilities.rs

### Rust Functions

#### `insert_sorted`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `insert`
  - `clone`
  - `push_back`

#### `is_core_module_path`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `and_then`
  - `file_stem`
  - `to_str`
  - `starts_with`
  - `starts_with`

#### `is_layer_violation`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `layer_prefix_value`
  - `layer_prefix_value`

#### `is_mmsb_main`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Private
- **Calls:**
  - `unwrap_or`
  - `map`
  - `and_then`
  - `file_name`
  - `to_str`

#### `layer_adheres`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `layer_prefix_value`
  - `layer_prefix_value`

#### `layer_prefix_value`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Private
- **Calls:**
  - `chars`
  - `String::new`
  - `next`
  - `is_ascii_digit`
  - `push`
  - `is_empty`
  - `ok`
  - `parse`

#### `layer_rank_map`

- **File:** src/010_layer_utilities.rs:0
- **Visibility:** Private
- **Calls:**
  - `HashMap::new`
  - `enumerate`
  - `iter`
  - `insert`
  - `clone`

## Layer: 020_gather_rust_files.rs

### Rust Functions

#### `gather_rust_files`

- **File:** src/020_gather_rust_files.rs:0
- **Visibility:** Public
- **Calls:**
  - `resolve_source_root`
  - `collect`
  - `map`
  - `filter`
  - `filter`
  - `filter_map`
  - `filter_entry`
  - `into_iter`
  - `WalkDir::new`
  - `depth`
  - `is_dir`
  - `file_type`
  - `allow_analysis_dir`
  - `path`
  - `ok`
  - `map_or`
  - `extension`
  - `path`
  - `unwrap_or`
  - `strip_prefix`
  - `path`
  - `path`
  - `count`
  - `components`
  - `starts_with`
  - `path`
  - `join`
  - `into_path`

## Layer: 030_is_cfg_test_item.rs

### Rust Functions

#### `is_cfg_test_item`

- **File:** src/030_is_cfg_test_item.rs:0
- **Visibility:** Public
- **Calls:**
  - `any`
  - `iter`
  - `item_attrs`
  - `is_ident`
  - `path`
  - `parse_nested_meta`
  - `is_ident`
  - `Ok`
  - `is_ident`
  - `parse_nested_meta`
  - `is_ident`
  - `Ok`
  - `Ok`

## Layer: 040_refactor_constraints.rs

### Rust Functions

#### `generate_constraints`

- **File:** src/040_refactor_constraints.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `filter_map`
  - `iter`

## Layer: 120_cluster_006.rs

### Rust Functions

#### `layer_prefix_value`

- **File:** src/120_cluster_006.rs:0
- **Visibility:** Public
- **Calls:**
  - `chars`
  - `String::new`
  - `next`
  - `is_ascii_digit`
  - `push`
  - `is_empty`
  - `ok`
  - `parse`

## Layer: 170_layer_utilities.rs

### Rust Functions

#### `main`

- **File:** src/170_layer_utilities.rs:0
- **Visibility:** Public
- **Calls:**
  - `Args::parse`
  - `canonicalize`
  - `join`
  - `std::env::current_dir`
  - `unwrap_or_else`
  - `canonicalize`
  - `join`
  - `std::env::current_dir`
  - `join`
  - `unwrap`
  - `std::env::current_dir`
  - `ok`
  - `std::fs::create_dir_all`
  - `unwrap_or`
  - `canonicalize`
  - `run_analysis`

## Layer: 190_conscience_graph.rs

### Rust Functions

#### `generate_conscience_map`

- **File:** src/190_conscience_graph.rs:0
- **Visibility:** Public
- **Calls:**
  - `String::new`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `HashMap::new`
  - `push`
  - `or_default`
  - `entry`
  - `clone`
  - `len`
  - `count`
  - `filter`
  - `values`
  - `any`
  - `iter`
  - `is_blocking`
  - `push_str`
  - `push_str`
  - `collect`
  - `into_iter`
  - `sort_by_key`
  - `count`
  - `filter`
  - `iter`
  - `is_blocking`
  - `push_str`
  - `push_str`
  - `count`
  - `filter`
  - `iter`
  - `is_blocking`
  - `len`
  - `push_str`
  - `first`
  - `is_empty`
  - `push_str`
  - `filter`
  - `iter`
  - `is_blocking`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `push_str`
  - `std::fs::write`
  - `Ok`

## Layer: 230_dead_code_attribute_parser.rs

### Rust Functions

#### `marker_from_str`

- **File:** src/230_dead_code_attribute_parser.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_str`
  - `to_ascii_lowercase`

## Layer: 330_markdown_report.rs

### Rust Functions

#### `generate_canonical_name`

- **File:** src/330_markdown_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `unwrap_or`
  - `and_then`
  - `file_stem`
  - `to_str`
  - `unwrap_or`
  - `and_then`
  - `extension`
  - `to_str`
  - `strip_numeric_prefix`
  - `is_empty`

## Layer: 340_main.rs

### Rust Functions

#### `main`

- **File:** src/340_main.rs:0
- **Visibility:** Private
- **Calls:**
  - `collect`
  - `std::env::args`
  - `len`
  - `agent_cli::run_agent_cli`
  - `crate::layer_utilities::main`

## Layer: 350_agent_cli.rs

### Rust Functions

#### `list_invariants`

- **File:** src/350_agent_cli.rs:0
- **Visibility:** Private
- **Calls:**
  - `load_invariants`
  - `collect`
  - `cloned`
  - `filter`
  - `iter`
  - `is_blocking`
  - `serde_json::to_string_pretty`
  - `Ok`

#### `load_invariants`

- **File:** src/350_agent_cli.rs:0
- **Visibility:** Private
- **Calls:**
  - `std::fs::read_to_string`
  - `Ok`
  - `serde_json::from_str`

## Layer: 380_dead_code_doc_comment_parser.rs

### Rust Functions

#### `item_attrs`

- **File:** src/380_dead_code_doc_comment_parser.rs:0
- **Visibility:** Crate

#### `item_name`

- **File:** src/380_dead_code_doc_comment_parser.rs:0
- **Visibility:** Crate
- **Calls:**
  - `Some`
  - `to_string`
  - `Some`
  - `to_string`
  - `Some`
  - `to_string`
  - `Some`
  - `to_string`
  - `Some`
  - `to_string`

#### `merge_doc_intent`

- **File:** src/380_dead_code_doc_comment_parser.rs:0
- **Visibility:** Public
- **Calls:**
  - `IntentMap::new`
  - `HashSet::new`
  - `insert`
  - `push`
  - `or_default`
  - `entry`
  - `clone`

## Layer: 390_dead_code_call_graph.rs

### Rust Functions

#### `is_reachable`

- **File:** src/390_dead_code_call_graph.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_empty`
  - `contains`
  - `compute_reachability`

#### `is_test_only`

- **File:** src/390_dead_code_call_graph.rs:0
- **Visibility:** Public
- **Calls:**
  - `contains`
  - `build_reverse_call_graph`
  - `get`
  - `is_empty`
  - `all`
  - `iter`
  - `contains`

## Layer: 400_dead_code_intent.rs

### Rust Functions

#### `merge_intent_sources`

- **File:** src/400_dead_code_intent.rs:0
- **Visibility:** Public
- **Calls:**
  - `IntentMap::new`
  - `extend`
  - `or_default`
  - `entry`
  - `extend`
  - `or_default`
  - `entry`
  - `extend`
  - `or_default`
  - `entry`

## Layer: 410_dead_code_test_boundaries.rs

### Rust Functions

#### `has_test_attr`

- **File:** src/410_dead_code_test_boundaries.rs:0
- **Visibility:** Crate
- **Calls:**
  - `any`
  - `iter`
  - `path`
  - `is_ident`
  - `map`
  - `last`
  - `to_string`

#### `item_attrs`

- **File:** src/410_dead_code_test_boundaries.rs:0
- **Visibility:** Private

## Layer: 420_dead_code_entrypoints.rs

### Rust Functions

#### `is_public_api`

- **File:** src/420_dead_code_entrypoints.rs:0
- **Visibility:** Public
- **Calls:**
  - `contains`

## Layer: 430_dead_code_classifier.rs

### Rust Functions

#### `is_reachable`

- **File:** src/430_dead_code_classifier.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_empty`
  - `crate::dead_code_call_graph::compute_reachability`
  - `contains`

## Layer: 500_dead_code_cli.rs

### Rust Functions

#### `is_test_path`

- **File:** src/500_dead_code_cli.rs:0
- **Visibility:** Crate
- **Calls:**
  - `any`
  - `components`
  - `unwrap_or`
  - `to_str`
  - `as_os_str`

#### `merge_intent_map`

- **File:** src/500_dead_code_cli.rs:0
- **Visibility:** Crate
- **Calls:**
  - `extend`
  - `or_default`
  - `entry`

## Layer: 520_dead_code_policy.rs

### Rust Functions

#### `load_policy`

- **File:** src/520_dead_code_policy.rs:0
- **Visibility:** Public
- **Calls:**
  - `std::fs::read_to_string`
  - `Ok`
  - `parse_policy`
  - `unwrap_or`
  - `parent`

## Layer: 640_correction_intelligence_report.rs

### Rust Functions

#### `generate_admission_preflight`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `evaluate_move_admission`
  - `push`
  - `clone`
  - `clone`
  - `clone`
  - `count`
  - `filter`
  - `iter`
  - `len`
  - `saturating_sub`
  - `len`
  - `clone`
  - `clone`
  - `clone`

#### `generate_correction_plan`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `action_symbol`
  - `push`
  - `action_module_path`
  - `action_refs`
  - `push`
  - `action_refs`
  - `push`
  - `clone`
  - `clone`
  - `clone`
  - `action_symbol`
  - `push`
  - `to_string`
  - `action_target_layer`
  - `action_function`
  - `push`
  - `action_function`
  - `action_target_layer`
  - `push`
  - `action_visibility`
  - `starts_with`
  - `push`
  - `to_string`
  - `push`
  - `push`
  - `unwrap_or`
  - `max`
  - `map`
  - `iter`
  - `action_id`
  - `to_vec`
  - `average_confidence`
  - `estimate_fix_time`
  - `len`

#### `generate_intelligence_report`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `predict_violations`
  - `fill_prediction_confidence`
  - `generate_correction_plan`
  - `augment_path_coherence_strategies`
  - `plan_verification_scope`
  - `build_rollback_criteria`
  - `estimate_impact`
  - `clone`
  - `push`
  - `push`
  - `push`
  - `push`
  - `compute_summary`
  - `to_string`
  - `to_rfc3339`
  - `chrono::Utc::now`
  - `clone`
  - `len`

#### `generate_phase2_cluster_slice`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Public
- **Calls:**
  - `parse_phase2_cluster_plan`
  - `ok_or_else`
  - `get`
  - `saturating_sub`
  - `std::io::Error::new`
  - `is_empty`
  - `Err`
  - `std::io::Error::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `Vec::new`
  - `clone`
  - `clone`
  - `clone`
  - `Some`
  - `to_string`
  - `display`
  - `generate_correction_plan`
  - `plan_verification_scope`
  - `build_rollback_criteria`
  - `clone`
  - `to_string`
  - `push`
  - `push`
  - `push`
  - `push`
  - `compute_summary`
  - `Ok`
  - `to_string`
  - `to_rfc3339`
  - `chrono::Utc::now`
  - `to_path_buf`
  - `len`

#### `is_function_signature_line`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `trim_start`
  - `contains`
  - `any`
  - `iter`
  - `contains`

#### `is_identifier_candidate`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `unwrap_or`
  - `map`
  - `next`
  - `chars`
  - `is_numeric`
  - `to_lowercase`
  - `contains`
  - `as_str`
  - `contains`
  - `as_str`

#### `is_test_attribute_line`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `starts_with`
  - `to_ascii_lowercase`
  - `contains`
  - `contains`

#### `is_test_scoped_function`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `lines`
  - `trim_start`
  - `starts_with`
  - `starts_with`
  - `contains`
  - `push`
  - `is_test_attribute_line`
  - `is_function_signature_line`
  - `is_empty`
  - `Ok`
  - `Ok`
  - `starts_with`
  - `is_empty`
  - `starts_with`
  - `count`
  - `filter`
  - `chars`
  - `count`
  - `filter`
  - `chars`
  - `last`
  - `pop`
  - `Ok`

#### `module_name_from_path`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Private
- **Calls:**
  - `and_then`
  - `file_stem`
  - `to_str`
  - `to_string`
  - `and_then`
  - `and_then`
  - `parent`
  - `file_name`
  - `to_str`
  - `to_string`
  - `Some`
  - `crate::cluster_010::normalize_module_name`

#### `move_violates_invariant`

- **File:** src/640_correction_intelligence_report.rs:0
- **Visibility:** Crate

## Layer: root

### Rust Functions

#### `generate_artifact`

- **File:** src/admission_composition_artifact.rs:0
- **Visibility:** Public
- **Calls:**
  - `to_rfc3339`
  - `chrono::Utc::now`
  - `project_invariants_touched`
  - `collect`
  - `map`
  - `iter`
  - `clone`
  - `project_conflict_reason`
  - `project_state`
  - `clone`
  - `to_string`
  - `len`
  - `to_string`

