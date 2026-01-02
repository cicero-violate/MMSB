# Invariant Analysis Report

Generated: 2026-01-01 02:49:50

## Summary

- **Total Invariants**: 1064
- **Proven**: 811 (76.2%)
- **Empirical**: 226
- **Heuristic**: 27 (2.5%) ⚠️ LOW CONFIDENCE
- **Violations**: 0

### By Kind

- **Structural**: 834
- **Semantic**: 230
- **Delta**: 0
- **Path-Intersection**: 0

## Proven Invariants (Mechanical Truth)

These invariants are mathematically proven from graph structure and should **always block refactorings**.

### collect_use_tree_idents

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_identifier_candidate

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### parse_phase2_cluster_plan

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### topo_sort_within

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### layer_constrained_sort

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 1 assignment is proven from call graph

### cyclomatic_complexity

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### predict_violations

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### write_plan_markdown

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Layer 1 assignment is proven from call graph

### list_invariants

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 1 assignment is proven from call graph

### parallel_build_file_dag

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/280_file_ordering.rs
- **Description**: Layer 0 assignment is proven from call graph

### find_reference_files

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### simulate_action

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### classify_tier

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/550_tier_classifier.rs
- **Description**: Layer 0 assignment is proven from call graph

### topological_sort

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### run_dead_code_pipeline

- **Type**: Structural: LayerFixed { layer: 4 }
- **File**: src/170_layer_utilities.rs
- **Description**: Layer 4 assignment is proven from call graph

### export_constraints_json

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Layer 0 assignment is proven from call graph

### compose_into_state

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/composition_rule.rs
- **Description**: Layer 1 assignment is proven from call graph

### compute_summary

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### compute_confidence

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/560_confidence_scorer.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_test_scoped_function

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### structural_layer_value

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### action_refs

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_public_api

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Layer 0 assignment is proven from call graph

### planned_directory_intent

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Layer 1 assignment is proven from call graph

### detect_latent_markers

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Layer 0 assignment is proven from call graph

### resolve_module_name

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 2 assignment is proven from call graph

### action_symbol

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_conscience_map

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/190_conscience_graph.rs
- **Description**: Layer 0 assignment is proven from call graph

### order_directories

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Layer 1 assignment is proven from call graph

### collect_julia_dependencies

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_reachable

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/430_dead_code_classifier.rs
- **Description**: Layer 0 assignment is proven from call graph

### marker_from_str

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 0 assignment is proven from call graph

### parse_bool

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Layer 0 assignment is proven from call graph

### find_test_callers

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Layer 1 assignment is proven from call graph

### path_common_prefix_len

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_directory_moves

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### contains_tools

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 0 assignment is proven from call graph

### check_planned_directory

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Layer 0 assignment is proven from call graph

### load_policy

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Layer 2 assignment is proven from call graph

### sort_structural_items

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 3 assignment is proven from call graph

### filter_path_coherence_report

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### find_element_file

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### topo_sort

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### write_intelligence_outputs_at

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Layer 2 assignment is proven from call graph

### check_conflicts

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/composition_rule.rs
- **Description**: Layer 1 assignment is proven from call graph

### generate_artifact

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 2 assignment is proven from call graph

### layer_rank_map

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### extract_layer

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/200_action_validator.rs
- **Description**: Layer 0 assignment is proven from call graph

### parse_list

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Layer 0 assignment is proven from call graph

### resolve_source_root

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/170_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### write_structural_batches

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### adjacency_from_edges

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### write_artifact

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 0 assignment is proven from call graph

### plan_verification_scope

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### collect_rust_dependencies

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### common_root

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Layer 0 assignment is proven from call graph

### show_stats

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 1 assignment is proven from call graph

### node_style

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### project_state

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 1 assignment is proven from call graph

### is_test_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Layer 0 assignment is proven from call graph

### parse_mmsb_latent_attr

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 2 assignment is proven from call graph

### assign_confidence

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/440_dead_code_confidence.rs
- **Description**: Layer 0 assignment is proven from call graph

### scan_file_attributes

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 2 assignment is proven from call graph

### is_cfg_test_item

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/030_is_cfg_test_item.rs
- **Description**: Layer 1 assignment is proven from call graph

### resolve_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_latent_attrs

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 1 assignment is proven from call graph

### default_confidence

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### affected_files

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### evaluate_move_admission

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 3 assignment is proven from call graph

### order_rust_files_by_dependency

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 1 assignment is proven from call graph

### collect_entrypoints

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Layer 1 assignment is proven from call graph

### extract_julia_dependencies

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 3 assignment is proven from call graph

### ordered_by_name

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_function_signature_line

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_naming_warnings

- **Type**: Structural: LayerFixed { layer: 7 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 7 assignment is proven from call graph

### generate_correction_plan

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### generate_phase2_cluster_slice

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### build_file_layers

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 1 assignment is proven from call graph

### project_invariants_touched

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_result

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 2 assignment is proven from call graph

### emit_verification_policy

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/630_verification_policy_emitter.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_exports

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Layer 1 assignment is proven from call graph

### merge_doc_intent

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Layer 0 assignment is proven from call graph

### naming_score_for_file

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### main

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/340_main.rs
- **Description**: Layer 3 assignment is proven from call graph

### build_call_graph

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Layer 0 assignment is proven from call graph

### action_target_layer

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### estimate_fix_time

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### action_module

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### merge_intent_sources

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Layer 0 assignment is proven from call graph

### function_in_impl_block

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### rust_entry_paths

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### layer_prefix_value

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Layer 0 assignment is proven from call graph

### export_program_cfg_to_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_invariants_touched

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/composition_rule.rs
- **Description**: Layer 0 assignment is proven from call graph

### extract_dependencies

- **Type**: Structural: LayerFixed { layer: 4 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 4 assignment is proven from call graph

### resolve_required_layer_path

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### detect_intent_signals

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 3 assignment is proven from call graph

### compare_path_components

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### validate_action

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/200_action_validator.rs
- **Description**: Layer 1 assignment is proven from call graph

### extract_critical_tests

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### compute_move_metrics

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### top_items

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Layer 0 assignment is proven from call graph

### average_confidence

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_file_dependency_graph

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 1 assignment is proven from call graph

### detect_violations

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### treat_public_as_entrypoint

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Layer 0 assignment is proven from call graph

### export_complete_program_dot

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_directory_dag

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 1 assignment is proven from call graph

### strip_numeric_prefix

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_module_map

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 0 assignment is proven from call graph

### action_function

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### serialize_correction_plan

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Layer 0 assignment is proven from call graph

### plan_options

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Layer 0 assignment is proven from call graph

### compare_dir_layers

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### build_rollback_criteria

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### detect_layer

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### gather_rust_files

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/020_gather_rust_files.rs
- **Description**: Layer 1 assignment is proven from call graph

### build_basic_report

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### check_action

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 1 assignment is proven from call graph

### compute_reachability

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Layer 0 assignment is proven from call graph

### find_callers

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### structural_cmp

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 2 assignment is proven from call graph

### extract_doc_markers

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Layer 1 assignment is proven from call graph

### write_summary_markdown

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Layer 1 assignment is proven from call graph

### run_analysis

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 2 assignment is proven from call graph

### collect_directory_files

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### run_agent_cli

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 2 assignment is proven from call graph

### module_name_from_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_symbols

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Layer 0 assignment is proven from call graph

### detect_cycles

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### order_julia_files_by_dependency

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 2 assignment is proven from call graph

### julia_entry_paths

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### augment_path_coherence_strategies

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### is_test_attribute_line

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### read_artifact

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 0 assignment is proven from call graph

### fill_prediction_confidence

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### normalize_module_name

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 0 assignment is proven from call graph

### layer_adheres

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### is_mmsb_main

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_move_items

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### has_test_attr

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Layer 0 assignment is proven from call graph

### scan_doc_comments

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/211_dead_code_doc_comment_scanner.rs
- **Description**: Layer 2 assignment is proven from call graph

### write_intelligence_outputs

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Layer 3 assignment is proven from call graph

### move_violates_invariant

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### find_function_definition_candidates

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### detect_test_symbols

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 2 assignment is proven from call graph

### parse_cluster_members

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_intelligence_report

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 3 assignment is proven from call graph

### write_outputs

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/470_dead_code_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### extract_attribute_value

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 0 assignment is proven from call graph

### load_invariants

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 0 assignment is proven from call graph

### parse_policy

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Layer 1 assignment is proven from call graph

### action_module_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_test_only

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Layer 1 assignment is proven from call graph

### function_signature_found

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### recommend_action

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/450_dead_code_actions.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_layer_violation

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### estimate_impact

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### collect_cluster_plans

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 2 assignment is proven from call graph

### insert_sorted

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_constraints

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/040_refactor_constraints.rs
- **Description**: Layer 0 assignment is proven from call graph

### reason_for_category

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Layer 0 assignment is proven from call graph

### query_function

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Layer 1 assignment is proven from call graph

### extract_identifiers

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### compress_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### filter_visibility_report

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### write_report

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### project_conflict_reason

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Layer 0 assignment is proven from call graph

### is_core_module_path

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_dependency_map

- **Type**: Structural: LayerFixed { layer: 5 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 5 assignment is proven from call graph

### detect_test_modules

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 2 assignment is proven from call graph

### classify_symbol

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/040_classify_symbol.rs
- **Description**: Layer 2 assignment is proven from call graph

### build_reverse_call_graph

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_admission_preflight

- **Type**: Structural: LayerFixed { layer: 4 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 4 assignment is proven from call graph

### scan_intent_tags

- **Type**: Structural: LayerFixed { layer: 3 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Layer 3 assignment is proven from call graph

### build_directory_entry_map

- **Type**: Structural: LayerFixed { layer: 6 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 6 assignment is proven from call graph

### merge_intent_map

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_entries

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### collect_roots

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/100_dependency.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_file_dag

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_canonical_name

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### write_admission_preflight_report

- **Type**: Structural: LayerFixed { layer: 5 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 5 assignment is proven from call graph

### build_module_root_map

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 1 assignment is proven from call graph

### write_cluster_batches

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### serialize_correction_plans

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Layer 1 assignment is proven from call graph

### build_report

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### build_state

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### extract_rust_dependencies

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 2 assignment is proven from call graph

### resolve_module

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Layer 1 assignment is proven from call graph

### gather_julia_files

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 0 assignment is proven from call graph

### should_exclude_from_analysis

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Layer 0 assignment is proven from call graph

### symbol_exists

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### estimate_verification_time

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### compose_batch

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/composition_rule.rs
- **Description**: Layer 2 assignment is proven from call graph

### extract_function_block_from_contents

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 1 assignment is proven from call graph

### filter_dead_code_elements

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Layer 2 assignment is proven from call graph

### cluster_target_path

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### find_private_dependencies

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 2 assignment is proven from call graph

### compute_cohesion_score

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Layer 0 assignment is proven from call graph

### allow_analysis_dir

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/170_layer_utilities.rs
- **Description**: Layer 0 assignment is proven from call graph

### analyze_file_ordering

- **Type**: Structural: LayerFixed { layer: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Layer 2 assignment is proven from call graph

### item_name

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Layer 0 assignment is proven from call graph

### export_json

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Layer 0 assignment is proven from call graph

### calculate_quality_delta

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### action_visibility

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Layer 0 assignment is proven from call graph

### detect_layer_violation

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Layer 1 assignment is proven from call graph

### collect_excluded_symbols

- **Type**: Structural: LayerFixed { layer: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Layer 1 assignment is proven from call graph

### item_attrs

- **Type**: Structural: LayerFixed { layer: 0 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Layer 0 assignment is proven from call graph

### generate_artifact

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 3 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=0, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 3

### write_artifact

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### read_artifact

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### project_invariants_touched

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### project_conflict_reason

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### project_state

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### compose_batch

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/composition_rule.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### check_conflicts

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/composition_rule.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### compose_into_state

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/composition_rule.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### collect_invariants_touched

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/composition_rule.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### build_directory_entry_map

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 9 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=9 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 9

### collect_naming_warnings

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### layer_constrained_sort

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### topo_sort_within

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### detect_layer

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### rust_entry_paths

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### collect_rust_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### order_rust_files_by_dependency

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 3 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 3

### collect_julia_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### julia_entry_paths

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### build_file_layers

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### gather_julia_files

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### topological_sort

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### ordered_by_name

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=4, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 0

### build_entries

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### analyze_file_ordering

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 7 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=7 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 7

### naming_score_for_file

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### detect_cycles

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### detect_violations

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### export_complete_program_dot

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### order_julia_files_by_dependency

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### run_analysis

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 5 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Degree: in=1, out=5 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 5

### build_result

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 6 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=6 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 6

### adjacency_from_edges

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### topo_sort

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### layer_rank_map

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### insert_sorted

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_mmsb_main

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### is_layer_violation

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### layer_prefix_value

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### compare_dir_layers

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### compare_path_components

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### layer_adheres

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### structural_layer_value

- **Type**: Structural: DegreeStable { in_degree: 6, out_degree: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=6, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 6
  - Out-degree: 1

### detect_layer_violation

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### parse_cluster_members

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_core_module_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### cluster_target_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### collect_cluster_plans

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### node_style

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### cyclomatic_complexity

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### structural_cmp

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 4 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=1, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 4

### sort_structural_items

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 3 }
- **File**: src/010_layer_utilities.rs
- **Description**: Degree: in=0, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 3

### gather_rust_files

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 2 }
- **File**: src/020_gather_rust_files.rs
- **Description**: Degree: in=2, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 2

### is_cfg_test_item

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/030_is_cfg_test_item.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### classify_symbol

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/040_classify_symbol.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### generate_constraints

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/040_refactor_constraints.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### normalize_module_name

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### resolve_module

- **Type**: Structural: DegreeStable { in_degree: 5, out_degree: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=5, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 5
  - Out-degree: 1

### contains_tools

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### build_module_root_map

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### extract_rust_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### extract_julia_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 5 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=1, out=5 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 5

### resolve_module_name

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=4, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 1

### build_dependency_map

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### extract_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/060_module_resolution.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### build_module_map

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### resolve_path

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### build_directory_dag

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### build_file_dependency_graph

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### export_program_cfg_to_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### build_file_dag

- **Type**: Structural: DegreeStable { in_degree: 3, out_degree: 0 }
- **File**: src/080_cluster_011.rs
- **Description**: Degree: in=3, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 3
  - Out-degree: 0

### collect_roots

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/100_dependency.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### layer_prefix_value

- **Type**: Structural: DegreeStable { in_degree: 13, out_degree: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=13, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 13
  - Out-degree: 0

### order_directories

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### common_root

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### strip_numeric_prefix

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### compute_cohesion_score

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/120_cluster_006.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### resolve_source_root

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/170_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### allow_analysis_dir

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/170_layer_utilities.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### main

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/340_main.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### run_dead_code_pipeline

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 17 }
- **File**: src/170_layer_utilities.rs
- **Description**: Degree: in=0, out=17 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 17

### export_json

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### export_constraints_json

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### generate_conscience_map

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/190_conscience_graph.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### extract_layer

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/200_action_validator.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### validate_action

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/200_action_validator.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### scan_doc_comments

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 3 }
- **File**: src/211_dead_code_doc_comment_scanner.rs
- **Description**: Degree: in=2, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 3

### parse_mmsb_latent_attr

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 3 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=2, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 3

### scan_file_attributes

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 3 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=0, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 3

### extract_attribute_value

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### collect_latent_attrs

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### marker_from_str

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### scan_intent_tags

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 4 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=0, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 4

### detect_intent_signals

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 5 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=1, out=5 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 5

### detect_test_modules

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### detect_test_symbols

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### extract_identifiers

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### parallel_build_file_dag

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/280_file_ordering.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### compress_path

- **Type**: Structural: DegreeStable { in_degree: 5, out_degree: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=5, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 5
  - Out-degree: 0

### collect_directory_files

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### path_common_prefix_len

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### compute_move_metrics

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### generate_canonical_name

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### collect_directory_moves

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### write_structural_batches

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 2 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=0, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 2

### write_cluster_batches

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### resolve_required_layer_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### collect_move_items

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 5 }
- **File**: src/330_markdown_report.rs
- **Description**: Degree: in=0, out=5 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 5

### main

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/340_main.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### run_agent_cli

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 4 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=0, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 4

### check_action

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### query_function

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### list_invariants

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### show_stats

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### load_invariants

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 0 }
- **File**: src/350_agent_cli.rs
- **Description**: Degree: in=4, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 0

### detect_latent_markers

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### merge_doc_intent

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### extract_doc_markers

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### item_name

- **Type**: Structural: DegreeStable { in_degree: 3, out_degree: 0 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Degree: in=3, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 3
  - Out-degree: 0

### item_attrs

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### build_call_graph

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### build_reverse_call_graph

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### compute_reachability

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_reachable

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/430_dead_code_classifier.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### is_test_only

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### check_planned_directory

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### merge_intent_sources

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### planned_directory_intent

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### collect_symbols

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### find_test_callers

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### has_test_attr

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### item_attrs

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 0 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Degree: in=4, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 0

### collect_entrypoints

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### collect_exports

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### is_public_api

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### collect_use_tree_idents

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### treat_public_as_entrypoint

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_reachable

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/430_dead_code_classifier.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### assign_confidence

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/440_dead_code_confidence.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### recommend_action

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/450_dead_code_actions.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### build_report

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### write_report

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### build_basic_report

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/470_dead_code_report.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### write_outputs

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 3 }
- **File**: src/470_dead_code_report.rs
- **Description**: Degree: in=1, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 3

### filter_dead_code_elements

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### should_exclude_from_analysis

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### collect_excluded_symbols

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### merge_intent_map

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### reason_for_category

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_test_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### load_policy

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### parse_policy

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 4 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Degree: in=1, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 4

### parse_list

- **Type**: Structural: DegreeStable { in_degree: 3, out_degree: 0 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Degree: in=3, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 3
  - Out-degree: 0

### parse_bool

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### write_summary_markdown

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### write_plan_markdown

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### top_items

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### plan_options

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### classify_tier

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/550_tier_classifier.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### compute_confidence

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/560_confidence_scorer.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### serialize_correction_plan

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### serialize_correction_plans

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### write_intelligence_outputs_at

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### write_intelligence_outputs

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### emit_verification_policy

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/630_verification_policy_emitter.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### build_state

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 0

### filter_path_coherence_report

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### write_admission_preflight_report

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### generate_admission_preflight

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### evaluate_move_admission

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 4 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 4

### find_private_dependencies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### extract_function_block_from_contents

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### extract_identifiers

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### is_identifier_candidate

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### find_function_definition_candidates

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### function_signature_found

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### is_function_signature_line

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=4, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 0

### function_in_impl_block

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### is_test_attribute_line

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### is_test_scoped_function

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### filter_visibility_report

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 1

### parse_phase2_cluster_plan

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### generate_phase2_cluster_slice

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 5 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=5 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 5

### augment_path_coherence_strategies

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### module_name_from_path

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### compute_summary

- **Type**: Structural: DegreeStable { in_degree: 4, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=4, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 4
  - Out-degree: 0

### fill_prediction_confidence

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### default_confidence

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### calculate_quality_delta

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### action_function

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### find_element_file

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### symbol_exists

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### move_violates_invariant

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### average_confidence

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### estimate_fix_time

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### action_symbol

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### action_module_path

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### action_refs

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### action_target_layer

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 0

### action_visibility

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### affected_files

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### action_module

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### estimate_verification_time

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### extract_critical_tests

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### find_callers

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### find_reference_files

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 1

### simulate_action

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 0 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=0 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 0

### predict_violations

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 4 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=4 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 4

### generate_correction_plan

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 12 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=12 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 12

### plan_verification_scope

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 3 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=3 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 3

### build_rollback_criteria

- **Type**: Structural: DegreeStable { in_degree: 2, out_degree: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=2, out=1 (proven from graph)
- **Evidence**:
  - In-degree: 2
  - Out-degree: 1

### estimate_impact

- **Type**: Structural: DegreeStable { in_degree: 1, out_degree: 2 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=1, out=2 (proven from graph)
- **Evidence**:
  - In-degree: 1
  - Out-degree: 2

### generate_intelligence_report

- **Type**: Structural: DegreeStable { in_degree: 0, out_degree: 8 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Degree: in=0, out=8 (proven from graph)
- **Evidence**:
  - In-degree: 0
  - Out-degree: 8

### generate_artifact

- **Type**: Structural: Root
- **File**: src/admission_composition_artifact.rs
- **Description**: Root node (called by no other functions)

### write_artifact

- **Type**: Structural: Leaf
- **File**: src/admission_composition_artifact.rs
- **Description**: Leaf node (calls no other functions)

### write_artifact

- **Type**: Structural: Root
- **File**: src/admission_composition_artifact.rs
- **Description**: Root node (called by no other functions)

### read_artifact

- **Type**: Structural: Leaf
- **File**: src/admission_composition_artifact.rs
- **Description**: Leaf node (calls no other functions)

### read_artifact

- **Type**: Structural: Root
- **File**: src/admission_composition_artifact.rs
- **Description**: Root node (called by no other functions)

### project_invariants_touched

- **Type**: Structural: Leaf
- **File**: src/admission_composition_artifact.rs
- **Description**: Leaf node (calls no other functions)

### project_conflict_reason

- **Type**: Structural: Leaf
- **File**: src/admission_composition_artifact.rs
- **Description**: Leaf node (calls no other functions)

### compose_batch

- **Type**: Structural: Root
- **File**: src/composition_rule.rs
- **Description**: Root node (called by no other functions)

### collect_invariants_touched

- **Type**: Structural: Leaf
- **File**: src/composition_rule.rs
- **Description**: Leaf node (calls no other functions)

### collect_naming_warnings

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### topo_sort_within

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### detect_layer

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### rust_entry_paths

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### collect_rust_dependencies

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### order_rust_files_by_dependency

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### collect_julia_dependencies

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### julia_entry_paths

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### julia_entry_paths

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### gather_julia_files

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### topological_sort

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### ordered_by_name

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### build_entries

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### analyze_file_ordering

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### naming_score_for_file

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### detect_cycles

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### detect_violations

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### export_complete_program_dot

- **Type**: Structural: Leaf
- **File**: src/000_dependency_analysis.rs
- **Description**: Leaf node (calls no other functions)

### export_complete_program_dot

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### order_julia_files_by_dependency

- **Type**: Structural: Root
- **File**: src/000_dependency_analysis.rs
- **Description**: Root node (called by no other functions)

### build_result

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### adjacency_from_edges

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### layer_rank_map

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### insert_sorted

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### is_mmsb_main

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### layer_prefix_value

- **Type**: Structural: Leaf
- **File**: src/120_cluster_006.rs
- **Description**: Leaf node (calls no other functions)

### layer_prefix_value

- **Type**: Structural: Root
- **File**: src/120_cluster_006.rs
- **Description**: Root node (called by no other functions)

### compare_dir_layers

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### compare_path_components

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### layer_adheres

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### detect_layer_violation

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### parse_cluster_members

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### is_core_module_path

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### collect_cluster_plans

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### node_style

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### node_style

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### cyclomatic_complexity

- **Type**: Structural: Leaf
- **File**: src/010_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### cyclomatic_complexity

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### sort_structural_items

- **Type**: Structural: Root
- **File**: src/010_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### generate_constraints

- **Type**: Structural: Leaf
- **File**: src/040_refactor_constraints.rs
- **Description**: Leaf node (calls no other functions)

### normalize_module_name

- **Type**: Structural: Leaf
- **File**: src/060_module_resolution.rs
- **Description**: Leaf node (calls no other functions)

### contains_tools

- **Type**: Structural: Leaf
- **File**: src/060_module_resolution.rs
- **Description**: Leaf node (calls no other functions)

### build_module_root_map

- **Type**: Structural: Root
- **File**: src/060_module_resolution.rs
- **Description**: Root node (called by no other functions)

### build_module_map

- **Type**: Structural: Leaf
- **File**: src/080_cluster_011.rs
- **Description**: Leaf node (calls no other functions)

### resolve_path

- **Type**: Structural: Leaf
- **File**: src/080_cluster_011.rs
- **Description**: Leaf node (calls no other functions)

### resolve_path

- **Type**: Structural: Root
- **File**: src/080_cluster_011.rs
- **Description**: Root node (called by no other functions)

### build_directory_dag

- **Type**: Structural: Root
- **File**: src/080_cluster_011.rs
- **Description**: Root node (called by no other functions)

### build_file_dependency_graph

- **Type**: Structural: Root
- **File**: src/080_cluster_011.rs
- **Description**: Root node (called by no other functions)

### export_program_cfg_to_path

- **Type**: Structural: Leaf
- **File**: src/080_cluster_011.rs
- **Description**: Leaf node (calls no other functions)

### build_file_dag

- **Type**: Structural: Leaf
- **File**: src/080_cluster_011.rs
- **Description**: Leaf node (calls no other functions)

### collect_roots

- **Type**: Structural: Leaf
- **File**: src/100_dependency.rs
- **Description**: Leaf node (calls no other functions)

### collect_roots

- **Type**: Structural: Root
- **File**: src/100_dependency.rs
- **Description**: Root node (called by no other functions)

### layer_prefix_value

- **Type**: Structural: Leaf
- **File**: src/120_cluster_006.rs
- **Description**: Leaf node (calls no other functions)

### order_directories

- **Type**: Structural: Root
- **File**: src/120_cluster_006.rs
- **Description**: Root node (called by no other functions)

### common_root

- **Type**: Structural: Leaf
- **File**: src/120_cluster_006.rs
- **Description**: Leaf node (calls no other functions)

### strip_numeric_prefix

- **Type**: Structural: Leaf
- **File**: src/120_cluster_006.rs
- **Description**: Leaf node (calls no other functions)

### compute_cohesion_score

- **Type**: Structural: Leaf
- **File**: src/120_cluster_006.rs
- **Description**: Leaf node (calls no other functions)

### compute_cohesion_score

- **Type**: Structural: Root
- **File**: src/120_cluster_006.rs
- **Description**: Root node (called by no other functions)

### resolve_source_root

- **Type**: Structural: Leaf
- **File**: src/170_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### allow_analysis_dir

- **Type**: Structural: Leaf
- **File**: src/170_layer_utilities.rs
- **Description**: Leaf node (calls no other functions)

### main

- **Type**: Structural: Leaf
- **File**: src/340_main.rs
- **Description**: Leaf node (calls no other functions)

### main

- **Type**: Structural: Root
- **File**: src/340_main.rs
- **Description**: Root node (called by no other functions)

### run_dead_code_pipeline

- **Type**: Structural: Root
- **File**: src/170_layer_utilities.rs
- **Description**: Root node (called by no other functions)

### export_json

- **Type**: Structural: Leaf
- **File**: src/180_invariant_reporter.rs
- **Description**: Leaf node (calls no other functions)

### export_json

- **Type**: Structural: Root
- **File**: src/180_invariant_reporter.rs
- **Description**: Root node (called by no other functions)

### export_constraints_json

- **Type**: Structural: Leaf
- **File**: src/180_invariant_reporter.rs
- **Description**: Leaf node (calls no other functions)

### export_constraints_json

- **Type**: Structural: Root
- **File**: src/180_invariant_reporter.rs
- **Description**: Root node (called by no other functions)

### generate_conscience_map

- **Type**: Structural: Leaf
- **File**: src/190_conscience_graph.rs
- **Description**: Leaf node (calls no other functions)

### generate_conscience_map

- **Type**: Structural: Root
- **File**: src/190_conscience_graph.rs
- **Description**: Root node (called by no other functions)

### extract_layer

- **Type**: Structural: Leaf
- **File**: src/200_action_validator.rs
- **Description**: Leaf node (calls no other functions)

### validate_action

- **Type**: Structural: Root
- **File**: src/200_action_validator.rs
- **Description**: Root node (called by no other functions)

### scan_file_attributes

- **Type**: Structural: Root
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Root node (called by no other functions)

### extract_attribute_value

- **Type**: Structural: Leaf
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Leaf node (calls no other functions)

### extract_attribute_value

- **Type**: Structural: Root
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Root node (called by no other functions)

### marker_from_str

- **Type**: Structural: Leaf
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Leaf node (calls no other functions)

### scan_intent_tags

- **Type**: Structural: Root
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Root node (called by no other functions)

### extract_identifiers

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### extract_identifiers

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### parallel_build_file_dag

- **Type**: Structural: Leaf
- **File**: src/280_file_ordering.rs
- **Description**: Leaf node (calls no other functions)

### parallel_build_file_dag

- **Type**: Structural: Root
- **File**: src/280_file_ordering.rs
- **Description**: Root node (called by no other functions)

### compress_path

- **Type**: Structural: Leaf
- **File**: src/330_markdown_report.rs
- **Description**: Leaf node (calls no other functions)

### collect_directory_files

- **Type**: Structural: Leaf
- **File**: src/330_markdown_report.rs
- **Description**: Leaf node (calls no other functions)

### path_common_prefix_len

- **Type**: Structural: Leaf
- **File**: src/330_markdown_report.rs
- **Description**: Leaf node (calls no other functions)

### compute_move_metrics

- **Type**: Structural: Leaf
- **File**: src/330_markdown_report.rs
- **Description**: Leaf node (calls no other functions)

### generate_canonical_name

- **Type**: Structural: Root
- **File**: src/330_markdown_report.rs
- **Description**: Root node (called by no other functions)

### collect_directory_moves

- **Type**: Structural: Root
- **File**: src/330_markdown_report.rs
- **Description**: Root node (called by no other functions)

### write_structural_batches

- **Type**: Structural: Root
- **File**: src/330_markdown_report.rs
- **Description**: Root node (called by no other functions)

### write_cluster_batches

- **Type**: Structural: Root
- **File**: src/330_markdown_report.rs
- **Description**: Root node (called by no other functions)

### collect_move_items

- **Type**: Structural: Root
- **File**: src/330_markdown_report.rs
- **Description**: Root node (called by no other functions)

### main

- **Type**: Structural: Root
- **File**: src/340_main.rs
- **Description**: Root node (called by no other functions)

### run_agent_cli

- **Type**: Structural: Root
- **File**: src/350_agent_cli.rs
- **Description**: Root node (called by no other functions)

### load_invariants

- **Type**: Structural: Leaf
- **File**: src/350_agent_cli.rs
- **Description**: Leaf node (calls no other functions)

### detect_latent_markers

- **Type**: Structural: Leaf
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Leaf node (calls no other functions)

### merge_doc_intent

- **Type**: Structural: Leaf
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Leaf node (calls no other functions)

### item_name

- **Type**: Structural: Leaf
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Leaf node (calls no other functions)

### item_attrs

- **Type**: Structural: Leaf
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Leaf node (calls no other functions)

### item_attrs

- **Type**: Structural: Root
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Root node (called by no other functions)

### build_call_graph

- **Type**: Structural: Leaf
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Leaf node (calls no other functions)

### build_reverse_call_graph

- **Type**: Structural: Leaf
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Leaf node (calls no other functions)

### compute_reachability

- **Type**: Structural: Leaf
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Leaf node (calls no other functions)

### is_reachable

- **Type**: Structural: Leaf
- **File**: src/430_dead_code_classifier.rs
- **Description**: Leaf node (calls no other functions)

### is_reachable

- **Type**: Structural: Root
- **File**: src/430_dead_code_classifier.rs
- **Description**: Root node (called by no other functions)

### check_planned_directory

- **Type**: Structural: Leaf
- **File**: src/400_dead_code_intent.rs
- **Description**: Leaf node (calls no other functions)

### merge_intent_sources

- **Type**: Structural: Leaf
- **File**: src/400_dead_code_intent.rs
- **Description**: Leaf node (calls no other functions)

### collect_symbols

- **Type**: Structural: Leaf
- **File**: src/400_dead_code_intent.rs
- **Description**: Leaf node (calls no other functions)

### find_test_callers

- **Type**: Structural: Root
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Root node (called by no other functions)

### has_test_attr

- **Type**: Structural: Leaf
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Leaf node (calls no other functions)

### item_attrs

- **Type**: Structural: Leaf
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Leaf node (calls no other functions)

### is_public_api

- **Type**: Structural: Leaf
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Leaf node (calls no other functions)

### collect_use_tree_idents

- **Type**: Structural: Leaf
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Leaf node (calls no other functions)

### treat_public_as_entrypoint

- **Type**: Structural: Leaf
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Leaf node (calls no other functions)

### assign_confidence

- **Type**: Structural: Leaf
- **File**: src/440_dead_code_confidence.rs
- **Description**: Leaf node (calls no other functions)

### recommend_action

- **Type**: Structural: Leaf
- **File**: src/450_dead_code_actions.rs
- **Description**: Leaf node (calls no other functions)

### build_report

- **Type**: Structural: Leaf
- **File**: src/470_dead_code_report.rs
- **Description**: Leaf node (calls no other functions)

### write_report

- **Type**: Structural: Leaf
- **File**: src/470_dead_code_report.rs
- **Description**: Leaf node (calls no other functions)

### build_basic_report

- **Type**: Structural: Leaf
- **File**: src/470_dead_code_report.rs
- **Description**: Leaf node (calls no other functions)

### build_basic_report

- **Type**: Structural: Root
- **File**: src/470_dead_code_report.rs
- **Description**: Root node (called by no other functions)

### filter_dead_code_elements

- **Type**: Structural: Root
- **File**: src/480_dead_code_filter.rs
- **Description**: Root node (called by no other functions)

### should_exclude_from_analysis

- **Type**: Structural: Leaf
- **File**: src/480_dead_code_filter.rs
- **Description**: Leaf node (calls no other functions)

### merge_intent_map

- **Type**: Structural: Leaf
- **File**: src/500_dead_code_cli.rs
- **Description**: Leaf node (calls no other functions)

### reason_for_category

- **Type**: Structural: Leaf
- **File**: src/500_dead_code_cli.rs
- **Description**: Leaf node (calls no other functions)

### is_test_path

- **Type**: Structural: Leaf
- **File**: src/500_dead_code_cli.rs
- **Description**: Leaf node (calls no other functions)

### load_policy

- **Type**: Structural: Root
- **File**: src/520_dead_code_policy.rs
- **Description**: Root node (called by no other functions)

### parse_list

- **Type**: Structural: Leaf
- **File**: src/520_dead_code_policy.rs
- **Description**: Leaf node (calls no other functions)

### parse_bool

- **Type**: Structural: Leaf
- **File**: src/520_dead_code_policy.rs
- **Description**: Leaf node (calls no other functions)

### top_items

- **Type**: Structural: Leaf
- **File**: src/540_dead_code_report_split.rs
- **Description**: Leaf node (calls no other functions)

### plan_options

- **Type**: Structural: Leaf
- **File**: src/540_dead_code_report_split.rs
- **Description**: Leaf node (calls no other functions)

### classify_tier

- **Type**: Structural: Leaf
- **File**: src/550_tier_classifier.rs
- **Description**: Leaf node (calls no other functions)

### classify_tier

- **Type**: Structural: Root
- **File**: src/550_tier_classifier.rs
- **Description**: Root node (called by no other functions)

### compute_confidence

- **Type**: Structural: Leaf
- **File**: src/560_confidence_scorer.rs
- **Description**: Leaf node (calls no other functions)

### compute_confidence

- **Type**: Structural: Root
- **File**: src/560_confidence_scorer.rs
- **Description**: Root node (called by no other functions)

### serialize_correction_plan

- **Type**: Structural: Leaf
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Leaf node (calls no other functions)

### write_intelligence_outputs

- **Type**: Structural: Root
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Root node (called by no other functions)

### emit_verification_policy

- **Type**: Structural: Leaf
- **File**: src/630_verification_policy_emitter.rs
- **Description**: Leaf node (calls no other functions)

### build_state

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### build_state

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### filter_path_coherence_report

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### write_admission_preflight_report

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### is_identifier_candidate

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### is_function_signature_line

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### is_test_attribute_line

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### filter_visibility_report

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### parse_phase2_cluster_plan

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### generate_phase2_cluster_slice

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### module_name_from_path

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### compute_summary

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### default_confidence

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### calculate_quality_delta

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_function

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### find_element_file

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### symbol_exists

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### move_violates_invariant

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### average_confidence

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### estimate_fix_time

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_symbol

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_module_path

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_refs

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_target_layer

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_visibility

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### affected_files

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### action_module

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### estimate_verification_time

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### extract_critical_tests

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### simulate_action

- **Type**: Structural: Leaf
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Leaf node (calls no other functions)

### generate_intelligence_report

- **Type**: Structural: Root
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Root node (called by no other functions)

### project_invariants_touched

- **Type**: Structural: SccMembership { scc_id: 0, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 0 (size: 1)

### project_state

- **Type**: Structural: SccMembership { scc_id: 1, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 1 (size: 1)

### project_conflict_reason

- **Type**: Structural: SccMembership { scc_id: 2, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 2 (size: 1)

### generate_artifact

- **Type**: Structural: SccMembership { scc_id: 3, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 3 (size: 1)

### write_artifact

- **Type**: Structural: SccMembership { scc_id: 4, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 4 (size: 1)

### read_artifact

- **Type**: Structural: SccMembership { scc_id: 5, scc_size: 1 }
- **File**: src/admission_composition_artifact.rs
- **Description**: Member of SCC 5 (size: 1)

### collect_invariants_touched

- **Type**: Structural: SccMembership { scc_id: 6, scc_size: 1 }
- **File**: src/composition_rule.rs
- **Description**: Member of SCC 6 (size: 1)

### compose_into_state

- **Type**: Structural: SccMembership { scc_id: 7, scc_size: 1 }
- **File**: src/composition_rule.rs
- **Description**: Member of SCC 7 (size: 1)

### check_conflicts

- **Type**: Structural: SccMembership { scc_id: 8, scc_size: 1 }
- **File**: src/composition_rule.rs
- **Description**: Member of SCC 8 (size: 1)

### compose_batch

- **Type**: Structural: SccMembership { scc_id: 9, scc_size: 1 }
- **File**: src/composition_rule.rs
- **Description**: Member of SCC 9 (size: 1)

### build_entries

- **Type**: Structural: SccMembership { scc_id: 10, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 10 (size: 1)

### ordered_by_name

- **Type**: Structural: SccMembership { scc_id: 11, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 11 (size: 1)

### topological_sort

- **Type**: Structural: SccMembership { scc_id: 12, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 12 (size: 1)

### topo_sort_within

- **Type**: Structural: SccMembership { scc_id: 13, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 13 (size: 1)

### layer_prefix_value

- **Type**: Structural: SccMembership { scc_id: 14, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 14 (size: 1)

### layer_constrained_sort

- **Type**: Structural: SccMembership { scc_id: 15, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 15 (size: 1)

### detect_cycles

- **Type**: Structural: SccMembership { scc_id: 16, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 16 (size: 1)

### build_file_dag

- **Type**: Structural: SccMembership { scc_id: 17, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 17 (size: 1)

### detect_layer

- **Type**: Structural: SccMembership { scc_id: 18, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 18 (size: 1)

### build_file_layers

- **Type**: Structural: SccMembership { scc_id: 19, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 19 (size: 1)

### normalize_module_name

- **Type**: Structural: SccMembership { scc_id: 20, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 20 (size: 1)

### resolve_module

- **Type**: Structural: SccMembership { scc_id: 21, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 21 (size: 1)

### resolve_module_name

- **Type**: Structural: SccMembership { scc_id: 22, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 22 (size: 1)

### extract_julia_dependencies

- **Type**: Structural: SccMembership { scc_id: 23, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 23 (size: 1)

### extract_rust_dependencies

- **Type**: Structural: SccMembership { scc_id: 24, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 24 (size: 1)

### extract_dependencies

- **Type**: Structural: SccMembership { scc_id: 25, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 25 (size: 1)

### build_dependency_map

- **Type**: Structural: SccMembership { scc_id: 26, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 26 (size: 1)

### build_directory_entry_map

- **Type**: Structural: SccMembership { scc_id: 27, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 27 (size: 1)

### naming_score_for_file

- **Type**: Structural: SccMembership { scc_id: 28, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 28 (size: 1)

### collect_naming_warnings

- **Type**: Structural: SccMembership { scc_id: 29, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 29 (size: 1)

### rust_entry_paths

- **Type**: Structural: SccMembership { scc_id: 30, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 30 (size: 1)

### collect_rust_dependencies

- **Type**: Structural: SccMembership { scc_id: 31, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 31 (size: 1)

### order_rust_files_by_dependency

- **Type**: Structural: SccMembership { scc_id: 32, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 32 (size: 1)

### collect_julia_dependencies

- **Type**: Structural: SccMembership { scc_id: 33, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 33 (size: 1)

### julia_entry_paths

- **Type**: Structural: SccMembership { scc_id: 34, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 34 (size: 1)

### gather_julia_files

- **Type**: Structural: SccMembership { scc_id: 35, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 35 (size: 1)

### detect_violations

- **Type**: Structural: SccMembership { scc_id: 36, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 36 (size: 1)

### analyze_file_ordering

- **Type**: Structural: SccMembership { scc_id: 37, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 37 (size: 1)

### export_complete_program_dot

- **Type**: Structural: SccMembership { scc_id: 38, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 38 (size: 1)

### order_julia_files_by_dependency

- **Type**: Structural: SccMembership { scc_id: 39, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 39 (size: 1)

### export_program_cfg_to_path

- **Type**: Structural: SccMembership { scc_id: 40, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 40 (size: 1)

### generate_constraints

- **Type**: Structural: SccMembership { scc_id: 41, scc_size: 1 }
- **File**: src/040_refactor_constraints.rs
- **Description**: Member of SCC 41 (size: 1)

### build_call_graph

- **Type**: Structural: SccMembership { scc_id: 42, scc_size: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Member of SCC 42 (size: 1)

### allow_analysis_dir

- **Type**: Structural: SccMembership { scc_id: 43, scc_size: 1 }
- **File**: src/170_layer_utilities.rs
- **Description**: Member of SCC 43 (size: 1)

### resolve_source_root

- **Type**: Structural: SccMembership { scc_id: 44, scc_size: 1 }
- **File**: src/170_layer_utilities.rs
- **Description**: Member of SCC 44 (size: 1)

### gather_rust_files

- **Type**: Structural: SccMembership { scc_id: 45, scc_size: 1 }
- **File**: src/020_gather_rust_files.rs
- **Description**: Member of SCC 45 (size: 1)

### run_analysis

- **Type**: Structural: SccMembership { scc_id: 46, scc_size: 1 }
- **File**: src/000_dependency_analysis.rs
- **Description**: Member of SCC 46 (size: 1)

### is_layer_violation

- **Type**: Structural: SccMembership { scc_id: 47, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 47 (size: 1)

### is_mmsb_main

- **Type**: Structural: SccMembership { scc_id: 48, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 48 (size: 1)

### layer_rank_map

- **Type**: Structural: SccMembership { scc_id: 49, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 49 (size: 1)

### insert_sorted

- **Type**: Structural: SccMembership { scc_id: 50, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 50 (size: 1)

### topo_sort

- **Type**: Structural: SccMembership { scc_id: 51, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 51 (size: 1)

### adjacency_from_edges

- **Type**: Structural: SccMembership { scc_id: 52, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 52 (size: 1)

### build_result

- **Type**: Structural: SccMembership { scc_id: 53, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 53 (size: 1)

### layer_prefix_value

- **Type**: Structural: SccMembership { scc_id: 54, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 54 (size: 1)

### compare_dir_layers

- **Type**: Structural: SccMembership { scc_id: 55, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 55 (size: 1)

### compare_path_components

- **Type**: Structural: SccMembership { scc_id: 56, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 56 (size: 1)

### layer_adheres

- **Type**: Structural: SccMembership { scc_id: 57, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 57 (size: 1)

### structural_layer_value

- **Type**: Structural: SccMembership { scc_id: 58, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 58 (size: 1)

### detect_layer_violation

- **Type**: Structural: SccMembership { scc_id: 59, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 59 (size: 1)

### parse_cluster_members

- **Type**: Structural: SccMembership { scc_id: 60, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 60 (size: 1)

### is_core_module_path

- **Type**: Structural: SccMembership { scc_id: 61, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 61 (size: 1)

### cluster_target_path

- **Type**: Structural: SccMembership { scc_id: 62, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 62 (size: 1)

### collect_cluster_plans

- **Type**: Structural: SccMembership { scc_id: 63, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 63 (size: 1)

### node_style

- **Type**: Structural: SccMembership { scc_id: 64, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 64 (size: 1)

### cyclomatic_complexity

- **Type**: Structural: SccMembership { scc_id: 65, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 65 (size: 1)

### structural_cmp

- **Type**: Structural: SccMembership { scc_id: 66, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 66 (size: 1)

### sort_structural_items

- **Type**: Structural: SccMembership { scc_id: 67, scc_size: 1 }
- **File**: src/010_layer_utilities.rs
- **Description**: Member of SCC 67 (size: 1)

### item_attrs

- **Type**: Structural: SccMembership { scc_id: 68, scc_size: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Member of SCC 68 (size: 1)

### is_cfg_test_item

- **Type**: Structural: SccMembership { scc_id: 69, scc_size: 1 }
- **File**: src/030_is_cfg_test_item.rs
- **Description**: Member of SCC 69 (size: 1)

### compute_reachability

- **Type**: Structural: SccMembership { scc_id: 70, scc_size: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Member of SCC 70 (size: 1)

### is_reachable

- **Type**: Structural: SccMembership { scc_id: 71, scc_size: 1 }
- **File**: src/430_dead_code_classifier.rs
- **Description**: Member of SCC 71 (size: 1)

### build_reverse_call_graph

- **Type**: Structural: SccMembership { scc_id: 72, scc_size: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Member of SCC 72 (size: 1)

### is_test_only

- **Type**: Structural: SccMembership { scc_id: 73, scc_size: 1 }
- **File**: src/390_dead_code_call_graph.rs
- **Description**: Member of SCC 73 (size: 1)

### classify_symbol

- **Type**: Structural: SccMembership { scc_id: 74, scc_size: 1 }
- **File**: src/040_classify_symbol.rs
- **Description**: Member of SCC 74 (size: 1)

### contains_tools

- **Type**: Structural: SccMembership { scc_id: 75, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 75 (size: 1)

### build_module_root_map

- **Type**: Structural: SccMembership { scc_id: 76, scc_size: 1 }
- **File**: src/060_module_resolution.rs
- **Description**: Member of SCC 76 (size: 1)

### build_module_map

- **Type**: Structural: SccMembership { scc_id: 77, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 77 (size: 1)

### resolve_path

- **Type**: Structural: SccMembership { scc_id: 78, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 78 (size: 1)

### build_directory_dag

- **Type**: Structural: SccMembership { scc_id: 79, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 79 (size: 1)

### build_file_dependency_graph

- **Type**: Structural: SccMembership { scc_id: 80, scc_size: 1 }
- **File**: src/080_cluster_011.rs
- **Description**: Member of SCC 80 (size: 1)

### collect_roots

- **Type**: Structural: SccMembership { scc_id: 81, scc_size: 1 }
- **File**: src/100_dependency.rs
- **Description**: Member of SCC 81 (size: 1)

### common_root

- **Type**: Structural: SccMembership { scc_id: 82, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 82 (size: 1)

### order_directories

- **Type**: Structural: SccMembership { scc_id: 83, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 83 (size: 1)

### strip_numeric_prefix

- **Type**: Structural: SccMembership { scc_id: 84, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 84 (size: 1)

### compute_cohesion_score

- **Type**: Structural: SccMembership { scc_id: 85, scc_size: 1 }
- **File**: src/120_cluster_006.rs
- **Description**: Member of SCC 85 (size: 1)

### main

- **Type**: Structural: SccMembership { scc_id: 86, scc_size: 1 }
- **File**: src/340_main.rs
- **Description**: Member of SCC 86 (size: 1)

### plan_options

- **Type**: Structural: SccMembership { scc_id: 87, scc_size: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Member of SCC 87 (size: 1)

### top_items

- **Type**: Structural: SccMembership { scc_id: 88, scc_size: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Member of SCC 88 (size: 1)

### write_plan_markdown

- **Type**: Structural: SccMembership { scc_id: 89, scc_size: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Member of SCC 89 (size: 1)

### write_summary_markdown

- **Type**: Structural: SccMembership { scc_id: 90, scc_size: 1 }
- **File**: src/540_dead_code_report_split.rs
- **Description**: Member of SCC 90 (size: 1)

### write_report

- **Type**: Structural: SccMembership { scc_id: 91, scc_size: 1 }
- **File**: src/470_dead_code_report.rs
- **Description**: Member of SCC 91 (size: 1)

### write_outputs

- **Type**: Structural: SccMembership { scc_id: 92, scc_size: 1 }
- **File**: src/470_dead_code_report.rs
- **Description**: Member of SCC 92 (size: 1)

### build_report

- **Type**: Structural: SccMembership { scc_id: 93, scc_size: 1 }
- **File**: src/470_dead_code_report.rs
- **Description**: Member of SCC 93 (size: 1)

### recommend_action

- **Type**: Structural: SccMembership { scc_id: 94, scc_size: 1 }
- **File**: src/450_dead_code_actions.rs
- **Description**: Member of SCC 94 (size: 1)

### is_public_api

- **Type**: Structural: SccMembership { scc_id: 95, scc_size: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Member of SCC 95 (size: 1)

### assign_confidence

- **Type**: Structural: SccMembership { scc_id: 96, scc_size: 1 }
- **File**: src/440_dead_code_confidence.rs
- **Description**: Member of SCC 96 (size: 1)

### reason_for_category

- **Type**: Structural: SccMembership { scc_id: 97, scc_size: 1 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Member of SCC 97 (size: 1)

### collect_use_tree_idents

- **Type**: Structural: SccMembership { scc_id: 98, scc_size: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Member of SCC 98 (size: 1)

### collect_exports

- **Type**: Structural: SccMembership { scc_id: 99, scc_size: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Member of SCC 99 (size: 1)

### treat_public_as_entrypoint

- **Type**: Structural: SccMembership { scc_id: 100, scc_size: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Member of SCC 100 (size: 1)

### collect_entrypoints

- **Type**: Structural: SccMembership { scc_id: 101, scc_size: 1 }
- **File**: src/420_dead_code_entrypoints.rs
- **Description**: Member of SCC 101 (size: 1)

### is_test_path

- **Type**: Structural: SccMembership { scc_id: 102, scc_size: 1 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Member of SCC 102 (size: 1)

### has_test_attr

- **Type**: Structural: SccMembership { scc_id: 103, scc_size: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Member of SCC 103 (size: 1)

### detect_test_symbols

- **Type**: Structural: SccMembership { scc_id: 104, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 104 (size: 1)

### detect_test_modules

- **Type**: Structural: SccMembership { scc_id: 105, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 105 (size: 1)

### merge_intent_map

- **Type**: Structural: SccMembership { scc_id: 106, scc_size: 1 }
- **File**: src/500_dead_code_cli.rs
- **Description**: Member of SCC 106 (size: 1)

### merge_intent_sources

- **Type**: Structural: SccMembership { scc_id: 107, scc_size: 1 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Member of SCC 107 (size: 1)

### collect_symbols

- **Type**: Structural: SccMembership { scc_id: 108, scc_size: 1 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Member of SCC 108 (size: 1)

### check_planned_directory

- **Type**: Structural: SccMembership { scc_id: 109, scc_size: 1 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Member of SCC 109 (size: 1)

### planned_directory_intent

- **Type**: Structural: SccMembership { scc_id: 110, scc_size: 1 }
- **File**: src/400_dead_code_intent.rs
- **Description**: Member of SCC 110 (size: 1)

### merge_doc_intent

- **Type**: Structural: SccMembership { scc_id: 111, scc_size: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Member of SCC 111 (size: 1)

### detect_latent_markers

- **Type**: Structural: SccMembership { scc_id: 112, scc_size: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Member of SCC 112 (size: 1)

### extract_doc_markers

- **Type**: Structural: SccMembership { scc_id: 113, scc_size: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Member of SCC 113 (size: 1)

### item_name

- **Type**: Structural: SccMembership { scc_id: 114, scc_size: 1 }
- **File**: src/380_dead_code_doc_comment_parser.rs
- **Description**: Member of SCC 114 (size: 1)

### scan_doc_comments

- **Type**: Structural: SccMembership { scc_id: 115, scc_size: 1 }
- **File**: src/211_dead_code_doc_comment_scanner.rs
- **Description**: Member of SCC 115 (size: 1)

### marker_from_str

- **Type**: Structural: SccMembership { scc_id: 116, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 116 (size: 1)

### collect_latent_attrs

- **Type**: Structural: SccMembership { scc_id: 117, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 117 (size: 1)

### parse_mmsb_latent_attr

- **Type**: Structural: SccMembership { scc_id: 118, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 118 (size: 1)

### detect_intent_signals

- **Type**: Structural: SccMembership { scc_id: 119, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 119 (size: 1)

### run_dead_code_pipeline

- **Type**: Structural: SccMembership { scc_id: 120, scc_size: 1 }
- **File**: src/170_layer_utilities.rs
- **Description**: Member of SCC 120 (size: 1)

### export_json

- **Type**: Structural: SccMembership { scc_id: 121, scc_size: 1 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Member of SCC 121 (size: 1)

### export_constraints_json

- **Type**: Structural: SccMembership { scc_id: 122, scc_size: 1 }
- **File**: src/180_invariant_reporter.rs
- **Description**: Member of SCC 122 (size: 1)

### generate_conscience_map

- **Type**: Structural: SccMembership { scc_id: 123, scc_size: 1 }
- **File**: src/190_conscience_graph.rs
- **Description**: Member of SCC 123 (size: 1)

### extract_layer

- **Type**: Structural: SccMembership { scc_id: 124, scc_size: 1 }
- **File**: src/200_action_validator.rs
- **Description**: Member of SCC 124 (size: 1)

### validate_action

- **Type**: Structural: SccMembership { scc_id: 125, scc_size: 1 }
- **File**: src/200_action_validator.rs
- **Description**: Member of SCC 125 (size: 1)

### scan_file_attributes

- **Type**: Structural: SccMembership { scc_id: 126, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 126 (size: 1)

### extract_attribute_value

- **Type**: Structural: SccMembership { scc_id: 127, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 127 (size: 1)

### scan_intent_tags

- **Type**: Structural: SccMembership { scc_id: 128, scc_size: 1 }
- **File**: src/230_dead_code_attribute_parser.rs
- **Description**: Member of SCC 128 (size: 1)

### extract_identifiers

- **Type**: Structural: SccMembership { scc_id: 129, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 129 (size: 1)

### parallel_build_file_dag

- **Type**: Structural: SccMembership { scc_id: 130, scc_size: 1 }
- **File**: src/280_file_ordering.rs
- **Description**: Member of SCC 130 (size: 1)

### compress_path

- **Type**: Structural: SccMembership { scc_id: 131, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 131 (size: 1)

### collect_directory_files

- **Type**: Structural: SccMembership { scc_id: 132, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 132 (size: 1)

### path_common_prefix_len

- **Type**: Structural: SccMembership { scc_id: 133, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 133 (size: 1)

### compute_move_metrics

- **Type**: Structural: SccMembership { scc_id: 134, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 134 (size: 1)

### generate_canonical_name

- **Type**: Structural: SccMembership { scc_id: 135, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 135 (size: 1)

### collect_directory_moves

- **Type**: Structural: SccMembership { scc_id: 136, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 136 (size: 1)

### write_structural_batches

- **Type**: Structural: SccMembership { scc_id: 137, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 137 (size: 1)

### write_cluster_batches

- **Type**: Structural: SccMembership { scc_id: 138, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 138 (size: 1)

### resolve_required_layer_path

- **Type**: Structural: SccMembership { scc_id: 139, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 139 (size: 1)

### collect_move_items

- **Type**: Structural: SccMembership { scc_id: 140, scc_size: 1 }
- **File**: src/330_markdown_report.rs
- **Description**: Member of SCC 140 (size: 1)

### main

- **Type**: Structural: SccMembership { scc_id: 141, scc_size: 1 }
- **File**: src/340_main.rs
- **Description**: Member of SCC 141 (size: 1)

### load_invariants

- **Type**: Structural: SccMembership { scc_id: 142, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 142 (size: 1)

### show_stats

- **Type**: Structural: SccMembership { scc_id: 143, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 143 (size: 1)

### list_invariants

- **Type**: Structural: SccMembership { scc_id: 144, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 144 (size: 1)

### query_function

- **Type**: Structural: SccMembership { scc_id: 145, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 145 (size: 1)

### check_action

- **Type**: Structural: SccMembership { scc_id: 146, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 146 (size: 1)

### run_agent_cli

- **Type**: Structural: SccMembership { scc_id: 147, scc_size: 1 }
- **File**: src/350_agent_cli.rs
- **Description**: Member of SCC 147 (size: 1)

### item_attrs

- **Type**: Structural: SccMembership { scc_id: 148, scc_size: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Member of SCC 148 (size: 1)

### is_reachable

- **Type**: Structural: SccMembership { scc_id: 149, scc_size: 1 }
- **File**: src/430_dead_code_classifier.rs
- **Description**: Member of SCC 149 (size: 1)

### find_test_callers

- **Type**: Structural: SccMembership { scc_id: 150, scc_size: 1 }
- **File**: src/410_dead_code_test_boundaries.rs
- **Description**: Member of SCC 150 (size: 1)

### build_basic_report

- **Type**: Structural: SccMembership { scc_id: 151, scc_size: 1 }
- **File**: src/470_dead_code_report.rs
- **Description**: Member of SCC 151 (size: 1)

### should_exclude_from_analysis

- **Type**: Structural: SccMembership { scc_id: 152, scc_size: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Member of SCC 152 (size: 1)

### collect_excluded_symbols

- **Type**: Structural: SccMembership { scc_id: 153, scc_size: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Member of SCC 153 (size: 1)

### filter_dead_code_elements

- **Type**: Structural: SccMembership { scc_id: 154, scc_size: 1 }
- **File**: src/480_dead_code_filter.rs
- **Description**: Member of SCC 154 (size: 1)

### parse_bool

- **Type**: Structural: SccMembership { scc_id: 155, scc_size: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Member of SCC 155 (size: 1)

### parse_list

- **Type**: Structural: SccMembership { scc_id: 156, scc_size: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Member of SCC 156 (size: 1)

### parse_policy

- **Type**: Structural: SccMembership { scc_id: 157, scc_size: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Member of SCC 157 (size: 1)

### load_policy

- **Type**: Structural: SccMembership { scc_id: 158, scc_size: 1 }
- **File**: src/520_dead_code_policy.rs
- **Description**: Member of SCC 158 (size: 1)

### classify_tier

- **Type**: Structural: SccMembership { scc_id: 159, scc_size: 1 }
- **File**: src/550_tier_classifier.rs
- **Description**: Member of SCC 159 (size: 1)

### compute_confidence

- **Type**: Structural: SccMembership { scc_id: 160, scc_size: 1 }
- **File**: src/560_confidence_scorer.rs
- **Description**: Member of SCC 160 (size: 1)

### serialize_correction_plan

- **Type**: Structural: SccMembership { scc_id: 161, scc_size: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Member of SCC 161 (size: 1)

### serialize_correction_plans

- **Type**: Structural: SccMembership { scc_id: 162, scc_size: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Member of SCC 162 (size: 1)

### emit_verification_policy

- **Type**: Structural: SccMembership { scc_id: 163, scc_size: 1 }
- **File**: src/630_verification_policy_emitter.rs
- **Description**: Member of SCC 163 (size: 1)

### write_intelligence_outputs_at

- **Type**: Structural: SccMembership { scc_id: 164, scc_size: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Member of SCC 164 (size: 1)

### write_intelligence_outputs

- **Type**: Structural: SccMembership { scc_id: 165, scc_size: 1 }
- **File**: src/620_correction_plan_serializer.rs
- **Description**: Member of SCC 165 (size: 1)

### build_state

- **Type**: Structural: SccMembership { scc_id: 166, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 166 (size: 1)

### compute_summary

- **Type**: Structural: SccMembership { scc_id: 167, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 167 (size: 1)

### filter_path_coherence_report

- **Type**: Structural: SccMembership { scc_id: 168, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 168 (size: 1)

### is_identifier_candidate

- **Type**: Structural: SccMembership { scc_id: 169, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 169 (size: 1)

### extract_identifiers

- **Type**: Structural: SccMembership { scc_id: 170, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 170 (size: 1)

### is_function_signature_line

- **Type**: Structural: SccMembership { scc_id: 171, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 171 (size: 1)

### extract_function_block_from_contents

- **Type**: Structural: SccMembership { scc_id: 172, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 172 (size: 1)

### find_private_dependencies

- **Type**: Structural: SccMembership { scc_id: 173, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 173 (size: 1)

### is_test_attribute_line

- **Type**: Structural: SccMembership { scc_id: 174, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 174 (size: 1)

### is_test_scoped_function

- **Type**: Structural: SccMembership { scc_id: 175, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 175 (size: 1)

### function_in_impl_block

- **Type**: Structural: SccMembership { scc_id: 176, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 176 (size: 1)

### function_signature_found

- **Type**: Structural: SccMembership { scc_id: 177, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 177 (size: 1)

### find_function_definition_candidates

- **Type**: Structural: SccMembership { scc_id: 178, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 178 (size: 1)

### evaluate_move_admission

- **Type**: Structural: SccMembership { scc_id: 179, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 179 (size: 1)

### generate_admission_preflight

- **Type**: Structural: SccMembership { scc_id: 180, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 180 (size: 1)

### write_admission_preflight_report

- **Type**: Structural: SccMembership { scc_id: 181, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 181 (size: 1)

### filter_visibility_report

- **Type**: Structural: SccMembership { scc_id: 182, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 182 (size: 1)

### parse_phase2_cluster_plan

- **Type**: Structural: SccMembership { scc_id: 183, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 183 (size: 1)

### extract_critical_tests

- **Type**: Structural: SccMembership { scc_id: 184, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 184 (size: 1)

### build_rollback_criteria

- **Type**: Structural: SccMembership { scc_id: 185, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 185 (size: 1)

### estimate_verification_time

- **Type**: Structural: SccMembership { scc_id: 186, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 186 (size: 1)

### action_module

- **Type**: Structural: SccMembership { scc_id: 187, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 187 (size: 1)

### affected_files

- **Type**: Structural: SccMembership { scc_id: 188, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 188 (size: 1)

### plan_verification_scope

- **Type**: Structural: SccMembership { scc_id: 189, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 189 (size: 1)

### estimate_fix_time

- **Type**: Structural: SccMembership { scc_id: 190, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 190 (size: 1)

### average_confidence

- **Type**: Structural: SccMembership { scc_id: 191, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 191 (size: 1)

### action_visibility

- **Type**: Structural: SccMembership { scc_id: 192, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 192 (size: 1)

### action_target_layer

- **Type**: Structural: SccMembership { scc_id: 193, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 193 (size: 1)

### action_function

- **Type**: Structural: SccMembership { scc_id: 194, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 194 (size: 1)

### action_symbol

- **Type**: Structural: SccMembership { scc_id: 195, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 195 (size: 1)

### action_refs

- **Type**: Structural: SccMembership { scc_id: 196, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 196 (size: 1)

### action_module_path

- **Type**: Structural: SccMembership { scc_id: 197, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 197 (size: 1)

### generate_correction_plan

- **Type**: Structural: SccMembership { scc_id: 198, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 198 (size: 1)

### generate_phase2_cluster_slice

- **Type**: Structural: SccMembership { scc_id: 199, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 199 (size: 1)

### module_name_from_path

- **Type**: Structural: SccMembership { scc_id: 200, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 200 (size: 1)

### augment_path_coherence_strategies

- **Type**: Structural: SccMembership { scc_id: 201, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 201 (size: 1)

### default_confidence

- **Type**: Structural: SccMembership { scc_id: 202, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 202 (size: 1)

### fill_prediction_confidence

- **Type**: Structural: SccMembership { scc_id: 203, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 203 (size: 1)

### calculate_quality_delta

- **Type**: Structural: SccMembership { scc_id: 204, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 204 (size: 1)

### find_element_file

- **Type**: Structural: SccMembership { scc_id: 205, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 205 (size: 1)

### symbol_exists

- **Type**: Structural: SccMembership { scc_id: 206, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 206 (size: 1)

### move_violates_invariant

- **Type**: Structural: SccMembership { scc_id: 207, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 207 (size: 1)

### find_callers

- **Type**: Structural: SccMembership { scc_id: 208, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 208 (size: 1)

### find_reference_files

- **Type**: Structural: SccMembership { scc_id: 209, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 209 (size: 1)

### simulate_action

- **Type**: Structural: SccMembership { scc_id: 210, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 210 (size: 1)

### predict_violations

- **Type**: Structural: SccMembership { scc_id: 211, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 211 (size: 1)

### estimate_impact

- **Type**: Structural: SccMembership { scc_id: 212, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 212 (size: 1)

### generate_intelligence_report

- **Type**: Structural: SccMembership { scc_id: 213, scc_size: 1 }
- **File**: src/640_correction_intelligence_report.rs
- **Description**: Member of SCC 213 (size: 1)

## Empirical Invariants (High Confidence)

These invariants were observed across multiple paths/samples and have high confidence.

### project_state

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### check_conflicts

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### compose_into_state

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### topo_sort

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### build_dependency_map

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### detect_test_modules

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### check_action

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### query_function

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### list_invariants

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### show_stats

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### extract_doc_markers

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### is_test_only

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### collect_entrypoints

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### collect_exports

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### collect_excluded_symbols

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### write_summary_markdown

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### serialize_correction_plans

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### generate_admission_preflight

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### extract_function_block_from_contents

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

### function_signature_found

- **Type**: Structural: Bridge
- **Strength**: EMPIRICAL (paths: 1)
- **Confidence**: 0.50
- **Description**: Potential bridge node in call graph

*... and 206 more*

## Heuristic Signals (Low Confidence - Review Required)

⚠️ **WARNING**: These are based on naming patterns and heuristics. They require manual verification and should **NOT block refactorings**.

- **is_mmsb_main**: Likely pure function (HEURISTIC - verify manually) (src/010_layer_utilities.rs)
- **is_layer_violation**: Likely pure function (HEURISTIC - verify manually) (src/010_layer_utilities.rs)
- **parse_cluster_members**: Likely pure function (HEURISTIC - verify manually) (src/010_layer_utilities.rs)
- **is_core_module_path**: Likely pure function (HEURISTIC - verify manually) (src/010_layer_utilities.rs)
- **is_cfg_test_item**: Likely pure function (HEURISTIC - verify manually) (src/030_is_cfg_test_item.rs)
- **compute_cohesion_score**: Likely pure function (HEURISTIC - verify manually) (src/120_cluster_006.rs)
- **parse_mmsb_latent_attr**: Likely pure function (HEURISTIC - verify manually) (src/230_dead_code_attribute_parser.rs)
- **compute_move_metrics**: Likely pure function (HEURISTIC - verify manually) (src/330_markdown_report.rs)
- **compute_reachability**: Likely pure function (HEURISTIC - verify manually) (src/390_dead_code_call_graph.rs)
- **is_reachable**: Likely pure function (HEURISTIC - verify manually) (src/390_dead_code_call_graph.rs)

*... and 17 more (see JSON export)*

