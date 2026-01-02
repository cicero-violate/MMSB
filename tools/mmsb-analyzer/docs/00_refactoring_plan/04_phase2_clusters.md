## Phase 2: Cluster Extraction

Action: create the listed cluster files and move the grouped functions.
Note: use the batches below to keep changes small.

- Create cluster file `src/420_dead_code_entrypoints.rs` with 16 functions (cohesion 0.68)
- Create cluster file `src/230_dead_code_attribute_parser.rs` with 10 functions (cohesion 0.67)
- Create cluster file `src/230_dead_code_attribute_parser.rs` with 11 functions (cohesion 0.35)
- Create cluster file `src/010_layer_utilities.rs` with 2 functions (cohesion 0.09)

```bash
touch "src/420_dead_code_entrypoints.rs"
touch "src/230_dead_code_attribute_parser.rs"
touch "src/230_dead_code_attribute_parser.rs"
touch "src/010_layer_utilities.rs"
```

### Phase 2 Tips

Action: apply these guidelines while executing Phase 2 batches.
Note: these are advisory, not checklist items.

- Extract clusters as a unit; avoid splitting a cluster across files.
- Prefer creating new files before moving functions to keep diffs small.
- After each batch, update imports and run tests to lock in behavior.

### Phase 2 Batches

Action: execute batches in order and verify after each batch.
Note: each batch creates or fills a cluster file.

#### Batch 1: target `src/420_dead_code_entrypoints.rs`

Action: move the listed functions into the target module.
Note: use the rg commands to locate definitions and callers.

- Cluster cohesion 0.68, 16 functions
- Move `gather_julia_files` from `src/000_dependency_analysis.rs`
- Move `run_dead_code_pipeline` from `src/170_layer_utilities.rs`
- Move `detect_intent_signals` from `src/230_dead_code_attribute_parser.rs`
- Move `detect_latent_markers` from `src/380_dead_code_doc_comment_parser.rs`
- Move `merge_doc_intent` from `src/380_dead_code_doc_comment_parser.rs`
- Move `build_call_graph` from `src/390_dead_code_call_graph.rs`
- Move `is_reachable` from `src/390_dead_code_call_graph.rs`
- Move `check_planned_directory` from `src/400_dead_code_intent.rs`
- Move `merge_intent_sources` from `src/400_dead_code_intent.rs`
- Move `is_reachable` from `src/430_dead_code_classifier.rs`
- Move `assign_confidence` from `src/440_dead_code_confidence.rs`
- Move `recommend_action` from `src/450_dead_code_actions.rs`
- Move `build_report` from `src/470_dead_code_report.rs`
- Move `merge_intent_map` from `src/500_dead_code_cli.rs`
- Move `reason_for_category` from `src/500_dead_code_cli.rs`
- Move `is_test_path` from `src/500_dead_code_cli.rs`
- Verification gate: `cargo test`

```bash
rg -n "gather_julia_files" "src/000_dependency_analysis.rs"
rg -n "gather_julia_files" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "run_dead_code_pipeline" "src/170_layer_utilities.rs"
rg -n "run_dead_code_pipeline" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "detect_intent_signals" "src/230_dead_code_attribute_parser.rs"
rg -n "detect_intent_signals" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "detect_latent_markers" "src/380_dead_code_doc_comment_parser.rs"
rg -n "detect_latent_markers" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "merge_doc_intent" "src/380_dead_code_doc_comment_parser.rs"
rg -n "merge_doc_intent" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "build_call_graph" "src/390_dead_code_call_graph.rs"
rg -n "build_call_graph" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "is_reachable" "src/390_dead_code_call_graph.rs"
rg -n "is_reachable" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "check_planned_directory" "src/400_dead_code_intent.rs"
rg -n "check_planned_directory" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "merge_intent_sources" "src/400_dead_code_intent.rs"
rg -n "merge_intent_sources" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "is_reachable" "src/430_dead_code_classifier.rs"
rg -n "is_reachable" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "assign_confidence" "src/440_dead_code_confidence.rs"
rg -n "assign_confidence" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "recommend_action" "src/450_dead_code_actions.rs"
rg -n "recommend_action" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "build_report" "src/470_dead_code_report.rs"
rg -n "build_report" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "merge_intent_map" "src/500_dead_code_cli.rs"
rg -n "merge_intent_map" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "reason_for_category" "src/500_dead_code_cli.rs"
rg -n "reason_for_category" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "is_test_path" "src/500_dead_code_cli.rs"
rg -n "is_test_path" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
```

#### Batch 2: target `src/230_dead_code_attribute_parser.rs`

Action: move the listed functions into the target module.
Note: use the rg commands to locate definitions and callers.

- Cluster cohesion 0.67, 10 functions
- Move `gather_rust_files` from `src/020_gather_rust_files.rs`
- Move `classify_symbol` from `src/040_classify_symbol.rs`
- Move `resolve_source_root` from `src/170_layer_utilities.rs`
- Move `allow_analysis_dir` from `src/170_layer_utilities.rs`
- Move `item_name` from `src/380_dead_code_doc_comment_parser.rs`
- Move `item_attrs` from `src/380_dead_code_doc_comment_parser.rs`
- Move `compute_reachability` from `src/390_dead_code_call_graph.rs`
- Move `is_test_only` from `src/390_dead_code_call_graph.rs`
- Move `collect_symbols` from `src/400_dead_code_intent.rs`
- Move `item_attrs` from `src/410_dead_code_test_boundaries.rs`
- Verification gate: `cargo test`

```bash
rg -n "gather_rust_files" "src/020_gather_rust_files.rs"
rg -n "gather_rust_files" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "classify_symbol" "src/040_classify_symbol.rs"
rg -n "classify_symbol" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "resolve_source_root" "src/170_layer_utilities.rs"
rg -n "resolve_source_root" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "allow_analysis_dir" "src/170_layer_utilities.rs"
rg -n "allow_analysis_dir" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "item_name" "src/380_dead_code_doc_comment_parser.rs"
rg -n "item_name" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "item_attrs" "src/380_dead_code_doc_comment_parser.rs"
rg -n "item_attrs" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "compute_reachability" "src/390_dead_code_call_graph.rs"
rg -n "compute_reachability" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "is_test_only" "src/390_dead_code_call_graph.rs"
rg -n "is_test_only" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "collect_symbols" "src/400_dead_code_intent.rs"
rg -n "collect_symbols" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "item_attrs" "src/410_dead_code_test_boundaries.rs"
rg -n "item_attrs" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
```

#### Batch 3: target `src/230_dead_code_attribute_parser.rs`

Action: move the listed functions into the target module.
Note: use the rg commands to locate definitions and callers.

- Cluster cohesion 0.35, 11 functions
- Move `run_analysis` from `src/000_dependency_analysis.rs`
- Move `is_cfg_test_item` from `src/030_is_cfg_test_item.rs`
- Move `generate_constraints` from `src/040_refactor_constraints.rs`
- Move `export_program_cfg_to_path` from `src/080_cluster_011.rs`
- Move `main` from `src/170_layer_utilities.rs`
- Move `scan_doc_comments` from `src/211_dead_code_doc_comment_scanner.rs`
- Move `extract_doc_markers` from `src/380_dead_code_doc_comment_parser.rs`
- Move `build_reverse_call_graph` from `src/390_dead_code_call_graph.rs`
- Move `planned_directory_intent` from `src/400_dead_code_intent.rs`
- Move `find_test_callers` from `src/410_dead_code_test_boundaries.rs`
- Move `has_test_attr` from `src/410_dead_code_test_boundaries.rs`
- Verification gate: `cargo test`

```bash
rg -n "run_analysis" "src/000_dependency_analysis.rs"
rg -n "run_analysis" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "is_cfg_test_item" "src/030_is_cfg_test_item.rs"
rg -n "is_cfg_test_item" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "generate_constraints" "src/040_refactor_constraints.rs"
rg -n "generate_constraints" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "export_program_cfg_to_path" "src/080_cluster_011.rs"
rg -n "export_program_cfg_to_path" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "main" "src/170_layer_utilities.rs"
rg -n "main" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "scan_doc_comments" "src/211_dead_code_doc_comment_scanner.rs"
rg -n "scan_doc_comments" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "extract_doc_markers" "src/380_dead_code_doc_comment_parser.rs"
rg -n "extract_doc_markers" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "build_reverse_call_graph" "src/390_dead_code_call_graph.rs"
rg -n "build_reverse_call_graph" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "planned_directory_intent" "src/400_dead_code_intent.rs"
rg -n "planned_directory_intent" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "find_test_callers" "src/410_dead_code_test_boundaries.rs"
rg -n "find_test_callers" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "has_test_attr" "src/410_dead_code_test_boundaries.rs"
rg -n "has_test_attr" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
```

#### Batch 4: target `src/010_layer_utilities.rs`

Action: move the listed functions into the target module.
Note: use the rg commands to locate definitions and callers.

- Cluster cohesion 0.09, 2 functions
- Move `layer_constrained_sort` from `src/000_dependency_analysis.rs`
- Move `topo_sort_within` from `src/000_dependency_analysis.rs`
- Verification gate: `cargo test`

```bash
rg -n "layer_constrained_sort" "src/000_dependency_analysis.rs"
rg -n "layer_constrained_sort" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
rg -n "topo_sort_within" "src/000_dependency_analysis.rs"
rg -n "topo_sort_within" "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tools/mmsb-analyzer"
```

