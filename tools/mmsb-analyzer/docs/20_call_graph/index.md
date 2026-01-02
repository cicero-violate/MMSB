# Call Graph Analysis

This document shows the **interprocedural call graph** - which functions call which other functions.

> **Note:** This is NOT a control flow graph (CFG). CFG shows intraprocedural control flow (branches, loops) within individual functions.

## Call Graph Statistics

- Total functions: 214
- Total function calls: 315
- Maximum call depth: 13
- Leaf functions (no outgoing calls): 98

## Call Graph Visualization

```mermaid
graph TD
    src_admission_composition_artifact_rs__generate_artifact["src/admission_composition_artifact.rs::generate_artifact"]
    src_admission_composition_artifact_rs__write_artifact["src/admission_composition_artifact.rs::write_artifact"]
    src_admission_composition_artifact_rs__read_artifact["src/admission_composition_artifact.rs::read_artifact"]
    src_admission_composition_artifact_rs__project_invariants_touched["src/admission_composition_artifact.rs::project_invariants_touched"]
    src_admission_composition_artifact_rs__project_conflict_reason["src/admission_composition_artifact.rs::project_conflict_reason"]
    src_admission_composition_artifact_rs__project_state["src/admission_composition_artifact.rs::project_state"]
    src_composition_rule_rs__compose_batch["src/composition_rule.rs::compose_batch"]
    src_composition_rule_rs__check_conflicts["src/composition_rule.rs::check_conflicts"]
    src_composition_rule_rs__compose_into_state["src/composition_rule.rs::compose_into_state"]
    src_composition_rule_rs__collect_invariants_touched["src/composition_rule.rs::collect_invariants_touched"]
    src_000_dependency_analysis_rs__build_directory_entry_map["src/000_dependency_analysis.rs::build_directory_entry_map"]
    src_000_dependency_analysis_rs__collect_naming_warnings["src/000_dependency_analysis.rs::collect_naming_warnings"]
    src_000_dependency_analysis_rs__layer_constrained_sort["src/000_dependency_analysis.rs::layer_constrained_sort"]
    src_000_dependency_analysis_rs__topo_sort_within["src/000_dependency_analysis.rs::topo_sort_within"]
    src_000_dependency_analysis_rs__detect_layer["src/000_dependency_analysis.rs::detect_layer"]
    src_000_dependency_analysis_rs__rust_entry_paths["src/000_dependency_analysis.rs::rust_entry_paths"]
    src_000_dependency_analysis_rs__collect_rust_dependencies["src/000_dependency_analysis.rs::collect_rust_dependencies"]
    src_000_dependency_analysis_rs__order_rust_files_by_dependency["src/000_dependency_analysis.rs::order_rust_files_by_dependency"]
    src_000_dependency_analysis_rs__collect_julia_dependencies["src/000_dependency_analysis.rs::collect_julia_dependencies"]
    src_000_dependency_analysis_rs__julia_entry_paths["src/000_dependency_analysis.rs::julia_entry_paths"]
    src_000_dependency_analysis_rs__build_file_layers["src/000_dependency_analysis.rs::build_file_layers"]
    src_000_dependency_analysis_rs__gather_julia_files["src/000_dependency_analysis.rs::gather_julia_files"]
    src_000_dependency_analysis_rs__topological_sort["src/000_dependency_analysis.rs::topological_sort"]
    src_000_dependency_analysis_rs__ordered_by_name["src/000_dependency_analysis.rs::ordered_by_name"]
    src_000_dependency_analysis_rs__build_entries["src/000_dependency_analysis.rs::build_entries"]
    src_000_dependency_analysis_rs__analyze_file_ordering["src/000_dependency_analysis.rs::analyze_file_ordering"]
    src_000_dependency_analysis_rs__naming_score_for_file["src/000_dependency_analysis.rs::naming_score_for_file"]
    src_000_dependency_analysis_rs__detect_cycles["src/000_dependency_analysis.rs::detect_cycles"]
    src_000_dependency_analysis_rs__detect_violations["src/000_dependency_analysis.rs::detect_violations"]
    src_000_dependency_analysis_rs__export_complete_program_dot["src/000_dependency_analysis.rs::export_complete_program_dot"]
    src_000_dependency_analysis_rs__order_julia_files_by_dependency["src/000_dependency_analysis.rs::order_julia_files_by_dependency"]
    src_000_dependency_analysis_rs__run_analysis["src/000_dependency_analysis.rs::run_analysis"]
    src_010_layer_utilities_rs__build_result["src/010_layer_utilities.rs::build_result"]
    src_010_layer_utilities_rs__adjacency_from_edges["src/010_layer_utilities.rs::adjacency_from_edges"]
    src_010_layer_utilities_rs__topo_sort["src/010_layer_utilities.rs::topo_sort"]
    src_010_layer_utilities_rs__layer_rank_map["src/010_layer_utilities.rs::layer_rank_map"]
    src_010_layer_utilities_rs__insert_sorted["src/010_layer_utilities.rs::insert_sorted"]
    src_010_layer_utilities_rs__is_mmsb_main["src/010_layer_utilities.rs::is_mmsb_main"]
    src_010_layer_utilities_rs__is_layer_violation["src/010_layer_utilities.rs::is_layer_violation"]
    src_010_layer_utilities_rs__layer_prefix_value["src/010_layer_utilities.rs::layer_prefix_value"]
    src_010_layer_utilities_rs__compare_dir_layers["src/010_layer_utilities.rs::compare_dir_layers"]
    src_010_layer_utilities_rs__compare_path_components["src/010_layer_utilities.rs::compare_path_components"]
    src_010_layer_utilities_rs__layer_adheres["src/010_layer_utilities.rs::layer_adheres"]
    src_010_layer_utilities_rs__structural_layer_value["src/010_layer_utilities.rs::structural_layer_value"]
    src_010_layer_utilities_rs__detect_layer_violation["src/010_layer_utilities.rs::detect_layer_violation"]
    src_010_layer_utilities_rs__parse_cluster_members["src/010_layer_utilities.rs::parse_cluster_members"]
    src_010_layer_utilities_rs__is_core_module_path["src/010_layer_utilities.rs::is_core_module_path"]
    src_010_layer_utilities_rs__cluster_target_path["src/010_layer_utilities.rs::cluster_target_path"]
    src_010_layer_utilities_rs__collect_cluster_plans["src/010_layer_utilities.rs::collect_cluster_plans"]
    src_010_layer_utilities_rs__node_style["src/010_layer_utilities.rs::node_style"]
    src_010_layer_utilities_rs__cyclomatic_complexity["src/010_layer_utilities.rs::cyclomatic_complexity"]
    src_010_layer_utilities_rs__structural_cmp["src/010_layer_utilities.rs::structural_cmp"]
    src_010_layer_utilities_rs__sort_structural_items["src/010_layer_utilities.rs::sort_structural_items"]
    src_020_gather_rust_files_rs__gather_rust_files["src/020_gather_rust_files.rs::gather_rust_files"]
    src_030_is_cfg_test_item_rs__is_cfg_test_item["src/030_is_cfg_test_item.rs::is_cfg_test_item"]
    src_040_classify_symbol_rs__classify_symbol["src/040_classify_symbol.rs::classify_symbol"]
    src_040_refactor_constraints_rs__generate_constraints["src/040_refactor_constraints.rs::generate_constraints"]
    src_060_module_resolution_rs__normalize_module_name["src/060_module_resolution.rs::normalize_module_name"]
    src_060_module_resolution_rs__resolve_module["src/060_module_resolution.rs::resolve_module"]
    src_060_module_resolution_rs__contains_tools["src/060_module_resolution.rs::contains_tools"]
    src_060_module_resolution_rs__build_module_root_map["src/060_module_resolution.rs::build_module_root_map"]
    src_060_module_resolution_rs__extract_rust_dependencies["src/060_module_resolution.rs::extract_rust_dependencies"]
    src_060_module_resolution_rs__extract_julia_dependencies["src/060_module_resolution.rs::extract_julia_dependencies"]
    src_060_module_resolution_rs__resolve_module_name["src/060_module_resolution.rs::resolve_module_name"]
    src_060_module_resolution_rs__build_dependency_map["src/060_module_resolution.rs::build_dependency_map"]
    src_060_module_resolution_rs__extract_dependencies["src/060_module_resolution.rs::extract_dependencies"]
    src_080_cluster_011_rs__build_module_map["src/080_cluster_011.rs::build_module_map"]
    src_080_cluster_011_rs__resolve_path["src/080_cluster_011.rs::resolve_path"]
    src_080_cluster_011_rs__build_directory_dag["src/080_cluster_011.rs::build_directory_dag"]
    src_080_cluster_011_rs__build_file_dependency_graph["src/080_cluster_011.rs::build_file_dependency_graph"]
    src_080_cluster_011_rs__export_program_cfg_to_path["src/080_cluster_011.rs::export_program_cfg_to_path"]
    src_080_cluster_011_rs__build_file_dag["src/080_cluster_011.rs::build_file_dag"]
    src_100_dependency_rs__collect_roots["src/100_dependency.rs::collect_roots"]
    src_120_cluster_006_rs__layer_prefix_value["src/120_cluster_006.rs::layer_prefix_value"]
    src_120_cluster_006_rs__order_directories["src/120_cluster_006.rs::order_directories"]
    src_120_cluster_006_rs__common_root["src/120_cluster_006.rs::common_root"]
    src_120_cluster_006_rs__strip_numeric_prefix["src/120_cluster_006.rs::strip_numeric_prefix"]
    src_120_cluster_006_rs__compute_cohesion_score["src/120_cluster_006.rs::compute_cohesion_score"]
    src_170_layer_utilities_rs__resolve_source_root["src/170_layer_utilities.rs::resolve_source_root"]
    src_170_layer_utilities_rs__allow_analysis_dir["src/170_layer_utilities.rs::allow_analysis_dir"]
    src_170_layer_utilities_rs__main["src/170_layer_utilities.rs::main"]
    src_170_layer_utilities_rs__run_dead_code_pipeline["src/170_layer_utilities.rs::run_dead_code_pipeline"]
    src_180_invariant_reporter_rs__export_json["src/180_invariant_reporter.rs::export_json"]
    src_180_invariant_reporter_rs__export_constraints_json["src/180_invariant_reporter.rs::export_constraints_json"]
    src_190_conscience_graph_rs__generate_conscience_map["src/190_conscience_graph.rs::generate_conscience_map"]
    src_200_action_validator_rs__extract_layer["src/200_action_validator.rs::extract_layer"]
    src_200_action_validator_rs__validate_action["src/200_action_validator.rs::validate_action"]
    src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments["src/211_dead_code_doc_comment_scanner.rs::scan_doc_comments"]
    src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr["src/230_dead_code_attribute_parser.rs::parse_mmsb_latent_attr"]
    src_230_dead_code_attribute_parser_rs__scan_file_attributes["src/230_dead_code_attribute_parser.rs::scan_file_attributes"]
    src_230_dead_code_attribute_parser_rs__extract_attribute_value["src/230_dead_code_attribute_parser.rs::extract_attribute_value"]
    src_230_dead_code_attribute_parser_rs__collect_latent_attrs["src/230_dead_code_attribute_parser.rs::collect_latent_attrs"]
    src_230_dead_code_attribute_parser_rs__marker_from_str["src/230_dead_code_attribute_parser.rs::marker_from_str"]
    src_230_dead_code_attribute_parser_rs__scan_intent_tags["src/230_dead_code_attribute_parser.rs::scan_intent_tags"]
    src_230_dead_code_attribute_parser_rs__detect_intent_signals["src/230_dead_code_attribute_parser.rs::detect_intent_signals"]
    src_230_dead_code_attribute_parser_rs__detect_test_modules["src/230_dead_code_attribute_parser.rs::detect_test_modules"]
    src_230_dead_code_attribute_parser_rs__detect_test_symbols["src/230_dead_code_attribute_parser.rs::detect_test_symbols"]
    src_250_cohesion_analyzer_rs__extract_identifiers["src/250_cohesion_analyzer.rs::extract_identifiers"]
    src_280_file_ordering_rs__parallel_build_file_dag["src/280_file_ordering.rs::parallel_build_file_dag"]
    src_330_markdown_report_rs__compress_path["src/330_markdown_report.rs::compress_path"]
    src_330_markdown_report_rs__collect_directory_files["src/330_markdown_report.rs::collect_directory_files"]
    src_330_markdown_report_rs__path_common_prefix_len["src/330_markdown_report.rs::path_common_prefix_len"]
    src_330_markdown_report_rs__compute_move_metrics["src/330_markdown_report.rs::compute_move_metrics"]
    src_330_markdown_report_rs__generate_canonical_name["src/330_markdown_report.rs::generate_canonical_name"]
    src_330_markdown_report_rs__collect_directory_moves["src/330_markdown_report.rs::collect_directory_moves"]
    src_330_markdown_report_rs__write_structural_batches["src/330_markdown_report.rs::write_structural_batches"]
    src_330_markdown_report_rs__write_cluster_batches["src/330_markdown_report.rs::write_cluster_batches"]
    src_330_markdown_report_rs__resolve_required_layer_path["src/330_markdown_report.rs::resolve_required_layer_path"]
    src_330_markdown_report_rs__collect_move_items["src/330_markdown_report.rs::collect_move_items"]
    src_340_main_rs__main["src/340_main.rs::main"]
    src_350_agent_cli_rs__run_agent_cli["src/350_agent_cli.rs::run_agent_cli"]
    src_350_agent_cli_rs__check_action["src/350_agent_cli.rs::check_action"]
    src_350_agent_cli_rs__query_function["src/350_agent_cli.rs::query_function"]
    src_350_agent_cli_rs__list_invariants["src/350_agent_cli.rs::list_invariants"]
    src_350_agent_cli_rs__show_stats["src/350_agent_cli.rs::show_stats"]
    src_350_agent_cli_rs__load_invariants["src/350_agent_cli.rs::load_invariants"]
    src_380_dead_code_doc_comment_parser_rs__detect_latent_markers["src/380_dead_code_doc_comment_parser.rs::detect_latent_markers"]
    src_380_dead_code_doc_comment_parser_rs__merge_doc_intent["src/380_dead_code_doc_comment_parser.rs::merge_doc_intent"]
    src_380_dead_code_doc_comment_parser_rs__extract_doc_markers["src/380_dead_code_doc_comment_parser.rs::extract_doc_markers"]
    src_380_dead_code_doc_comment_parser_rs__item_name["src/380_dead_code_doc_comment_parser.rs::item_name"]
    src_380_dead_code_doc_comment_parser_rs__item_attrs["src/380_dead_code_doc_comment_parser.rs::item_attrs"]
    src_390_dead_code_call_graph_rs__build_call_graph["src/390_dead_code_call_graph.rs::build_call_graph"]
    src_390_dead_code_call_graph_rs__build_reverse_call_graph["src/390_dead_code_call_graph.rs::build_reverse_call_graph"]
    src_390_dead_code_call_graph_rs__compute_reachability["src/390_dead_code_call_graph.rs::compute_reachability"]
    src_390_dead_code_call_graph_rs__is_reachable["src/390_dead_code_call_graph.rs::is_reachable"]
    src_390_dead_code_call_graph_rs__is_test_only["src/390_dead_code_call_graph.rs::is_test_only"]
    src_400_dead_code_intent_rs__check_planned_directory["src/400_dead_code_intent.rs::check_planned_directory"]
    src_400_dead_code_intent_rs__merge_intent_sources["src/400_dead_code_intent.rs::merge_intent_sources"]
    src_400_dead_code_intent_rs__planned_directory_intent["src/400_dead_code_intent.rs::planned_directory_intent"]
    src_400_dead_code_intent_rs__collect_symbols["src/400_dead_code_intent.rs::collect_symbols"]
    src_410_dead_code_test_boundaries_rs__find_test_callers["src/410_dead_code_test_boundaries.rs::find_test_callers"]
    src_410_dead_code_test_boundaries_rs__has_test_attr["src/410_dead_code_test_boundaries.rs::has_test_attr"]
    src_410_dead_code_test_boundaries_rs__item_attrs["src/410_dead_code_test_boundaries.rs::item_attrs"]
    src_420_dead_code_entrypoints_rs__collect_entrypoints["src/420_dead_code_entrypoints.rs::collect_entrypoints"]
    src_420_dead_code_entrypoints_rs__collect_exports["src/420_dead_code_entrypoints.rs::collect_exports"]
    src_420_dead_code_entrypoints_rs__is_public_api["src/420_dead_code_entrypoints.rs::is_public_api"]
    src_420_dead_code_entrypoints_rs__collect_use_tree_idents["src/420_dead_code_entrypoints.rs::collect_use_tree_idents"]
    src_420_dead_code_entrypoints_rs__treat_public_as_entrypoint["src/420_dead_code_entrypoints.rs::treat_public_as_entrypoint"]
    src_430_dead_code_classifier_rs__is_reachable["src/430_dead_code_classifier.rs::is_reachable"]
    src_440_dead_code_confidence_rs__assign_confidence["src/440_dead_code_confidence.rs::assign_confidence"]
    src_450_dead_code_actions_rs__recommend_action["src/450_dead_code_actions.rs::recommend_action"]
    src_470_dead_code_report_rs__build_report["src/470_dead_code_report.rs::build_report"]
    src_470_dead_code_report_rs__write_report["src/470_dead_code_report.rs::write_report"]
    src_470_dead_code_report_rs__build_basic_report["src/470_dead_code_report.rs::build_basic_report"]
    src_470_dead_code_report_rs__write_outputs["src/470_dead_code_report.rs::write_outputs"]
    src_480_dead_code_filter_rs__filter_dead_code_elements["src/480_dead_code_filter.rs::filter_dead_code_elements"]
    src_480_dead_code_filter_rs__should_exclude_from_analysis["src/480_dead_code_filter.rs::should_exclude_from_analysis"]
    src_480_dead_code_filter_rs__collect_excluded_symbols["src/480_dead_code_filter.rs::collect_excluded_symbols"]
    src_500_dead_code_cli_rs__merge_intent_map["src/500_dead_code_cli.rs::merge_intent_map"]
    src_500_dead_code_cli_rs__reason_for_category["src/500_dead_code_cli.rs::reason_for_category"]
    src_500_dead_code_cli_rs__is_test_path["src/500_dead_code_cli.rs::is_test_path"]
    src_520_dead_code_policy_rs__load_policy["src/520_dead_code_policy.rs::load_policy"]
    src_520_dead_code_policy_rs__parse_policy["src/520_dead_code_policy.rs::parse_policy"]
    src_520_dead_code_policy_rs__parse_list["src/520_dead_code_policy.rs::parse_list"]
    src_520_dead_code_policy_rs__parse_bool["src/520_dead_code_policy.rs::parse_bool"]
    src_540_dead_code_report_split_rs__write_summary_markdown["src/540_dead_code_report_split.rs::write_summary_markdown"]
    src_540_dead_code_report_split_rs__write_plan_markdown["src/540_dead_code_report_split.rs::write_plan_markdown"]
    src_540_dead_code_report_split_rs__top_items["src/540_dead_code_report_split.rs::top_items"]
    src_540_dead_code_report_split_rs__plan_options["src/540_dead_code_report_split.rs::plan_options"]
    src_550_tier_classifier_rs__classify_tier["src/550_tier_classifier.rs::classify_tier"]
    src_560_confidence_scorer_rs__compute_confidence["src/560_confidence_scorer.rs::compute_confidence"]
    src_620_correction_plan_serializer_rs__serialize_correction_plan["src/620_correction_plan_serializer.rs::serialize_correction_plan"]
    src_620_correction_plan_serializer_rs__serialize_correction_plans["src/620_correction_plan_serializer.rs::serialize_correction_plans"]
    src_620_correction_plan_serializer_rs__write_intelligence_outputs_at["src/620_correction_plan_serializer.rs::write_intelligence_outputs_at"]
    src_620_correction_plan_serializer_rs__write_intelligence_outputs["src/620_correction_plan_serializer.rs::write_intelligence_outputs"]
    src_630_verification_policy_emitter_rs__emit_verification_policy["src/630_verification_policy_emitter.rs::emit_verification_policy"]
    src_640_correction_intelligence_report_rs__build_state["src/640_correction_intelligence_report.rs::build_state"]
    src_640_correction_intelligence_report_rs__filter_path_coherence_report["src/640_correction_intelligence_report.rs::filter_path_coherence_report"]
    src_640_correction_intelligence_report_rs__write_admission_preflight_report["src/640_correction_intelligence_report.rs::write_admission_preflight_report"]
    src_640_correction_intelligence_report_rs__generate_admission_preflight["src/640_correction_intelligence_report.rs::generate_admission_preflight"]
    src_640_correction_intelligence_report_rs__evaluate_move_admission["src/640_correction_intelligence_report.rs::evaluate_move_admission"]
    src_640_correction_intelligence_report_rs__find_private_dependencies["src/640_correction_intelligence_report.rs::find_private_dependencies"]
    src_640_correction_intelligence_report_rs__extract_function_block_from_contents["src/640_correction_intelligence_report.rs::extract_function_block_from_contents"]
    src_640_correction_intelligence_report_rs__extract_identifiers["src/640_correction_intelligence_report.rs::extract_identifiers"]
    src_640_correction_intelligence_report_rs__is_identifier_candidate["src/640_correction_intelligence_report.rs::is_identifier_candidate"]
    src_640_correction_intelligence_report_rs__find_function_definition_candidates["src/640_correction_intelligence_report.rs::find_function_definition_candidates"]
    src_640_correction_intelligence_report_rs__function_signature_found["src/640_correction_intelligence_report.rs::function_signature_found"]
    src_640_correction_intelligence_report_rs__is_function_signature_line["src/640_correction_intelligence_report.rs::is_function_signature_line"]
    src_640_correction_intelligence_report_rs__function_in_impl_block["src/640_correction_intelligence_report.rs::function_in_impl_block"]
    src_640_correction_intelligence_report_rs__is_test_attribute_line["src/640_correction_intelligence_report.rs::is_test_attribute_line"]
    src_640_correction_intelligence_report_rs__is_test_scoped_function["src/640_correction_intelligence_report.rs::is_test_scoped_function"]
    src_640_correction_intelligence_report_rs__filter_visibility_report["src/640_correction_intelligence_report.rs::filter_visibility_report"]
    src_640_correction_intelligence_report_rs__parse_phase2_cluster_plan["src/640_correction_intelligence_report.rs::parse_phase2_cluster_plan"]
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice["src/640_correction_intelligence_report.rs::generate_phase2_cluster_slice"]
    src_640_correction_intelligence_report_rs__augment_path_coherence_strategies["src/640_correction_intelligence_report.rs::augment_path_coherence_strategies"]
    src_640_correction_intelligence_report_rs__module_name_from_path["src/640_correction_intelligence_report.rs::module_name_from_path"]
    src_640_correction_intelligence_report_rs__compute_summary["src/640_correction_intelligence_report.rs::compute_summary"]
    src_640_correction_intelligence_report_rs__fill_prediction_confidence["src/640_correction_intelligence_report.rs::fill_prediction_confidence"]
    src_640_correction_intelligence_report_rs__default_confidence["src/640_correction_intelligence_report.rs::default_confidence"]
    src_640_correction_intelligence_report_rs__calculate_quality_delta["src/640_correction_intelligence_report.rs::calculate_quality_delta"]
    src_640_correction_intelligence_report_rs__action_function["src/640_correction_intelligence_report.rs::action_function"]
    src_640_correction_intelligence_report_rs__find_element_file["src/640_correction_intelligence_report.rs::find_element_file"]
    src_640_correction_intelligence_report_rs__symbol_exists["src/640_correction_intelligence_report.rs::symbol_exists"]
    src_640_correction_intelligence_report_rs__move_violates_invariant["src/640_correction_intelligence_report.rs::move_violates_invariant"]
    src_640_correction_intelligence_report_rs__average_confidence["src/640_correction_intelligence_report.rs::average_confidence"]
    src_640_correction_intelligence_report_rs__estimate_fix_time["src/640_correction_intelligence_report.rs::estimate_fix_time"]
    src_640_correction_intelligence_report_rs__action_symbol["src/640_correction_intelligence_report.rs::action_symbol"]
    src_640_correction_intelligence_report_rs__action_module_path["src/640_correction_intelligence_report.rs::action_module_path"]
    src_640_correction_intelligence_report_rs__action_refs["src/640_correction_intelligence_report.rs::action_refs"]
    src_640_correction_intelligence_report_rs__action_target_layer["src/640_correction_intelligence_report.rs::action_target_layer"]
    src_640_correction_intelligence_report_rs__action_visibility["src/640_correction_intelligence_report.rs::action_visibility"]
    src_640_correction_intelligence_report_rs__affected_files["src/640_correction_intelligence_report.rs::affected_files"]
    src_640_correction_intelligence_report_rs__action_module["src/640_correction_intelligence_report.rs::action_module"]
    src_640_correction_intelligence_report_rs__estimate_verification_time["src/640_correction_intelligence_report.rs::estimate_verification_time"]
    src_640_correction_intelligence_report_rs__extract_critical_tests["src/640_correction_intelligence_report.rs::extract_critical_tests"]
    src_640_correction_intelligence_report_rs__find_callers["src/640_correction_intelligence_report.rs::find_callers"]
    src_640_correction_intelligence_report_rs__find_reference_files["src/640_correction_intelligence_report.rs::find_reference_files"]
    src_640_correction_intelligence_report_rs__simulate_action["src/640_correction_intelligence_report.rs::simulate_action"]
    src_640_correction_intelligence_report_rs__predict_violations["src/640_correction_intelligence_report.rs::predict_violations"]
    src_640_correction_intelligence_report_rs__generate_correction_plan["src/640_correction_intelligence_report.rs::generate_correction_plan"]
    src_640_correction_intelligence_report_rs__plan_verification_scope["src/640_correction_intelligence_report.rs::plan_verification_scope"]
    src_640_correction_intelligence_report_rs__build_rollback_criteria["src/640_correction_intelligence_report.rs::build_rollback_criteria"]
    src_640_correction_intelligence_report_rs__estimate_impact["src/640_correction_intelligence_report.rs::estimate_impact"]
    src_640_correction_intelligence_report_rs__generate_intelligence_report["src/640_correction_intelligence_report.rs::generate_intelligence_report"]
    src_admission_composition_artifact_rs__generate_artifact --> src_admission_composition_artifact_rs__project_invariants_touched
    src_admission_composition_artifact_rs__generate_artifact --> src_admission_composition_artifact_rs__project_conflict_reason
    src_admission_composition_artifact_rs__generate_artifact --> src_admission_composition_artifact_rs__project_state
    src_admission_composition_artifact_rs__project_state --> src_admission_composition_artifact_rs__project_invariants_touched
    src_composition_rule_rs__compose_batch --> src_composition_rule_rs__check_conflicts
    src_composition_rule_rs__compose_batch --> src_composition_rule_rs__compose_into_state
    src_composition_rule_rs__check_conflicts --> src_composition_rule_rs__collect_invariants_touched
    src_composition_rule_rs__compose_into_state --> src_composition_rule_rs__collect_invariants_touched
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_080_cluster_011_rs__build_module_map
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_060_module_resolution_rs__build_dependency_map
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__build_file_layers
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_080_cluster_011_rs__build_file_dag
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__detect_cycles
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__layer_constrained_sort
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__topological_sort
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__ordered_by_name
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__ordered_by_name
    src_000_dependency_analysis_rs__build_directory_entry_map --> src_000_dependency_analysis_rs__build_entries
    src_000_dependency_analysis_rs__collect_naming_warnings --> src_000_dependency_analysis_rs__build_directory_entry_map
    src_000_dependency_analysis_rs__collect_naming_warnings --> src_000_dependency_analysis_rs__naming_score_for_file
    src_000_dependency_analysis_rs__collect_naming_warnings --> src_000_dependency_analysis_rs__collect_naming_warnings
    src_000_dependency_analysis_rs__layer_constrained_sort --> src_010_layer_utilities_rs__layer_prefix_value
    src_000_dependency_analysis_rs__layer_constrained_sort --> src_120_cluster_006_rs__layer_prefix_value
    src_000_dependency_analysis_rs__layer_constrained_sort --> src_000_dependency_analysis_rs__topo_sort_within
    src_000_dependency_analysis_rs__rust_entry_paths --> src_170_layer_utilities_rs__resolve_source_root
    src_000_dependency_analysis_rs__order_rust_files_by_dependency --> src_060_module_resolution_rs__build_module_root_map
    src_000_dependency_analysis_rs__order_rust_files_by_dependency --> src_000_dependency_analysis_rs__rust_entry_paths
    src_000_dependency_analysis_rs__order_rust_files_by_dependency --> src_000_dependency_analysis_rs__detect_layer
    src_000_dependency_analysis_rs__order_rust_files_by_dependency --> src_000_dependency_analysis_rs__collect_rust_dependencies
    src_000_dependency_analysis_rs__order_rust_files_by_dependency --> src_010_layer_utilities_rs__build_result
    src_000_dependency_analysis_rs__julia_entry_paths --> src_170_layer_utilities_rs__resolve_source_root
    src_000_dependency_analysis_rs__build_file_layers --> src_000_dependency_analysis_rs__detect_layer
    src_000_dependency_analysis_rs__gather_julia_files --> src_170_layer_utilities_rs__resolve_source_root
    src_000_dependency_analysis_rs__gather_julia_files --> src_170_layer_utilities_rs__allow_analysis_dir
    src_000_dependency_analysis_rs__build_entries --> src_330_markdown_report_rs__generate_canonical_name
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_080_cluster_011_rs__build_module_map
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_060_module_resolution_rs__build_dependency_map
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__build_file_layers
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_120_cluster_006_rs__order_directories
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_080_cluster_011_rs__build_file_dag
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__detect_cycles
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__layer_constrained_sort
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__topological_sort
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__ordered_by_name
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__ordered_by_name
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__build_entries
    src_000_dependency_analysis_rs__analyze_file_ordering --> src_000_dependency_analysis_rs__detect_violations
    src_000_dependency_analysis_rs__export_complete_program_dot --> src_010_layer_utilities_rs__cyclomatic_complexity
    src_000_dependency_analysis_rs__export_complete_program_dot --> src_010_layer_utilities_rs__node_style
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_000_dependency_analysis_rs__julia_entry_paths
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_000_dependency_analysis_rs__detect_layer
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_000_dependency_analysis_rs__collect_julia_dependencies
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_000_dependency_analysis_rs__detect_layer
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_060_module_resolution_rs__resolve_module
    src_000_dependency_analysis_rs__order_julia_files_by_dependency --> src_010_layer_utilities_rs__build_result
    src_000_dependency_analysis_rs__run_analysis --> src_020_gather_rust_files_rs__gather_rust_files
    src_000_dependency_analysis_rs__run_analysis --> src_000_dependency_analysis_rs__order_rust_files_by_dependency
    src_000_dependency_analysis_rs__run_analysis --> src_000_dependency_analysis_rs__analyze_file_ordering
    src_000_dependency_analysis_rs__run_analysis --> src_000_dependency_analysis_rs__gather_julia_files
    src_000_dependency_analysis_rs__run_analysis --> src_000_dependency_analysis_rs__order_julia_files_by_dependency
    src_000_dependency_analysis_rs__run_analysis --> src_520_dead_code_policy_rs__load_policy
    src_000_dependency_analysis_rs__run_analysis --> src_170_layer_utilities_rs__run_dead_code_pipeline
    src_000_dependency_analysis_rs__run_analysis --> src_480_dead_code_filter_rs__filter_dead_code_elements
    src_000_dependency_analysis_rs__run_analysis --> src_390_dead_code_call_graph_rs__build_call_graph
    src_000_dependency_analysis_rs__run_analysis --> src_040_refactor_constraints_rs__generate_constraints
    src_000_dependency_analysis_rs__run_analysis --> src_080_cluster_011_rs__export_program_cfg_to_path
    src_000_dependency_analysis_rs__run_analysis --> src_180_invariant_reporter_rs__export_constraints_json
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__adjacency_from_edges
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__topo_sort
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__layer_rank_map
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__is_mmsb_main
    src_010_layer_utilities_rs__build_result --> src_170_layer_utilities_rs__main
    src_010_layer_utilities_rs__build_result --> src_340_main_rs__main
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__is_mmsb_main
    src_010_layer_utilities_rs__build_result --> src_170_layer_utilities_rs__main
    src_010_layer_utilities_rs__build_result --> src_340_main_rs__main
    src_010_layer_utilities_rs__build_result --> src_010_layer_utilities_rs__is_layer_violation
    src_010_layer_utilities_rs__topo_sort --> src_010_layer_utilities_rs__insert_sorted
    src_010_layer_utilities_rs__is_layer_violation --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__is_layer_violation --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__is_layer_violation --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__is_layer_violation --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_dir_layers --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_dir_layers --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_dir_layers --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_dir_layers --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_path_components --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_path_components --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_path_components --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__compare_path_components --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__layer_adheres --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__layer_adheres --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__layer_adheres --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__layer_adheres --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__structural_layer_value --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__structural_layer_value --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__detect_layer_violation --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__detect_layer_violation --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__detect_layer_violation --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__detect_layer_violation --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__cluster_target_path --> src_010_layer_utilities_rs__is_core_module_path
    src_010_layer_utilities_rs__cluster_target_path --> src_010_layer_utilities_rs__layer_prefix_value
    src_010_layer_utilities_rs__cluster_target_path --> src_120_cluster_006_rs__layer_prefix_value
    src_010_layer_utilities_rs__collect_cluster_plans --> src_010_layer_utilities_rs__parse_cluster_members
    src_010_layer_utilities_rs__collect_cluster_plans --> src_010_layer_utilities_rs__cluster_target_path
    src_010_layer_utilities_rs__structural_cmp --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__structural_cmp --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__structural_cmp --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__structural_cmp --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__sort_structural_items --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__sort_structural_items --> src_010_layer_utilities_rs__structural_layer_value
    src_010_layer_utilities_rs__sort_structural_items --> src_010_layer_utilities_rs__structural_cmp
    src_020_gather_rust_files_rs__gather_rust_files --> src_170_layer_utilities_rs__resolve_source_root
    src_020_gather_rust_files_rs__gather_rust_files --> src_170_layer_utilities_rs__allow_analysis_dir
    src_030_is_cfg_test_item_rs__is_cfg_test_item --> src_380_dead_code_doc_comment_parser_rs__item_attrs
    src_030_is_cfg_test_item_rs__is_cfg_test_item --> src_410_dead_code_test_boundaries_rs__item_attrs
    src_040_classify_symbol_rs__classify_symbol --> src_390_dead_code_call_graph_rs__is_test_only
    src_040_classify_symbol_rs__classify_symbol --> src_390_dead_code_call_graph_rs__is_reachable
    src_040_classify_symbol_rs__classify_symbol --> src_430_dead_code_classifier_rs__is_reachable
    src_060_module_resolution_rs__resolve_module --> src_060_module_resolution_rs__normalize_module_name
    src_060_module_resolution_rs__resolve_module --> src_080_cluster_011_rs__resolve_path
    src_060_module_resolution_rs__build_module_root_map --> src_060_module_resolution_rs__contains_tools
    src_060_module_resolution_rs__build_module_root_map --> src_060_module_resolution_rs__normalize_module_name
    src_060_module_resolution_rs__build_module_root_map --> src_000_dependency_analysis_rs__detect_layer
    src_060_module_resolution_rs__extract_rust_dependencies --> src_100_dependency_rs__collect_roots
    src_060_module_resolution_rs__extract_rust_dependencies --> src_060_module_resolution_rs__resolve_module
    src_060_module_resolution_rs__extract_rust_dependencies --> src_060_module_resolution_rs__resolve_module
    src_060_module_resolution_rs__extract_julia_dependencies --> src_060_module_resolution_rs__resolve_module
    src_060_module_resolution_rs__extract_julia_dependencies --> src_080_cluster_011_rs__resolve_path
    src_060_module_resolution_rs__extract_julia_dependencies --> src_060_module_resolution_rs__resolve_module_name
    src_060_module_resolution_rs__extract_julia_dependencies --> src_060_module_resolution_rs__resolve_module_name
    src_060_module_resolution_rs__extract_julia_dependencies --> src_060_module_resolution_rs__resolve_module_name
    src_060_module_resolution_rs__extract_julia_dependencies --> src_060_module_resolution_rs__resolve_module_name
    src_060_module_resolution_rs__resolve_module_name --> src_060_module_resolution_rs__resolve_module
    src_060_module_resolution_rs__build_dependency_map --> src_060_module_resolution_rs__extract_dependencies
    src_060_module_resolution_rs__extract_dependencies --> src_060_module_resolution_rs__extract_rust_dependencies
    src_060_module_resolution_rs__extract_dependencies --> src_060_module_resolution_rs__extract_julia_dependencies
    src_080_cluster_011_rs__build_module_map --> src_060_module_resolution_rs__normalize_module_name
    src_080_cluster_011_rs__build_module_map --> src_060_module_resolution_rs__normalize_module_name
    src_080_cluster_011_rs__resolve_path --> src_060_module_resolution_rs__normalize_module_name
    src_080_cluster_011_rs__build_directory_dag --> src_080_cluster_011_rs__build_module_map
    src_080_cluster_011_rs__build_directory_dag --> src_060_module_resolution_rs__build_dependency_map
    src_080_cluster_011_rs__build_directory_dag --> src_080_cluster_011_rs__build_file_dag
    src_080_cluster_011_rs__build_file_dependency_graph --> src_080_cluster_011_rs__build_module_map
    src_080_cluster_011_rs__build_file_dependency_graph --> src_060_module_resolution_rs__build_dependency_map
    src_080_cluster_011_rs__build_file_dependency_graph --> src_080_cluster_011_rs__build_file_dag
    src_080_cluster_011_rs__export_program_cfg_to_path --> src_000_dependency_analysis_rs__export_complete_program_dot
    src_100_dependency_rs__collect_roots --> src_100_dependency_rs__collect_roots
    src_100_dependency_rs__collect_roots --> src_100_dependency_rs__collect_roots
    src_120_cluster_006_rs__order_directories --> src_120_cluster_006_rs__common_root
    src_120_cluster_006_rs__order_directories --> src_010_layer_utilities_rs__compare_path_components
    src_120_cluster_006_rs__compute_cohesion_score --> src_010_layer_utilities_rs__layer_adheres
    src_170_layer_utilities_rs__main --> src_000_dependency_analysis_rs__run_analysis
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_020_gather_rust_files_rs__gather_rust_files
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_230_dead_code_attribute_parser_rs__detect_intent_signals
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_500_dead_code_cli_rs__merge_intent_map
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_230_dead_code_attribute_parser_rs__detect_test_modules
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_230_dead_code_attribute_parser_rs__detect_test_symbols
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_500_dead_code_cli_rs__is_test_path
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_390_dead_code_call_graph_rs__build_call_graph
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_420_dead_code_entrypoints_rs__collect_entrypoints
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_420_dead_code_entrypoints_rs__collect_exports
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_040_classify_symbol_rs__classify_symbol
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_390_dead_code_call_graph_rs__is_reachable
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_430_dead_code_classifier_rs__is_reachable
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_500_dead_code_cli_rs__reason_for_category
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_440_dead_code_confidence_rs__assign_confidence
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_420_dead_code_entrypoints_rs__is_public_api
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_450_dead_code_actions_rs__recommend_action
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_470_dead_code_report_rs__build_report
    src_170_layer_utilities_rs__run_dead_code_pipeline --> src_470_dead_code_report_rs__write_outputs
    src_200_action_validator_rs__validate_action --> src_200_action_validator_rs__extract_layer
    src_200_action_validator_rs__validate_action --> src_200_action_validator_rs__extract_layer
    src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments --> src_380_dead_code_doc_comment_parser_rs__item_name
    src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments --> src_380_dead_code_doc_comment_parser_rs__extract_doc_markers
    src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments --> src_380_dead_code_doc_comment_parser_rs__item_attrs
    src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments --> src_410_dead_code_test_boundaries_rs__item_attrs
    src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr --> src_380_dead_code_doc_comment_parser_rs__item_name
    src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr --> src_230_dead_code_attribute_parser_rs__collect_latent_attrs
    src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr --> src_380_dead_code_doc_comment_parser_rs__item_attrs
    src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr --> src_410_dead_code_test_boundaries_rs__item_attrs
    src_230_dead_code_attribute_parser_rs__scan_file_attributes --> src_380_dead_code_doc_comment_parser_rs__item_name
    src_230_dead_code_attribute_parser_rs__scan_file_attributes --> src_230_dead_code_attribute_parser_rs__collect_latent_attrs
    src_230_dead_code_attribute_parser_rs__scan_file_attributes --> src_380_dead_code_doc_comment_parser_rs__item_attrs
    src_230_dead_code_attribute_parser_rs__scan_file_attributes --> src_410_dead_code_test_boundaries_rs__item_attrs
    src_230_dead_code_attribute_parser_rs__collect_latent_attrs --> src_230_dead_code_attribute_parser_rs__marker_from_str
    src_230_dead_code_attribute_parser_rs__scan_intent_tags --> src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr
    src_230_dead_code_attribute_parser_rs__scan_intent_tags --> src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments
    src_230_dead_code_attribute_parser_rs__scan_intent_tags --> src_400_dead_code_intent_rs__check_planned_directory
    src_230_dead_code_attribute_parser_rs__scan_intent_tags --> src_400_dead_code_intent_rs__collect_symbols
    src_230_dead_code_attribute_parser_rs__detect_intent_signals --> src_230_dead_code_attribute_parser_rs__parse_mmsb_latent_attr
    src_230_dead_code_attribute_parser_rs__detect_intent_signals --> src_211_dead_code_doc_comment_scanner_rs__scan_doc_comments
    src_230_dead_code_attribute_parser_rs__detect_intent_signals --> src_380_dead_code_doc_comment_parser_rs__merge_doc_intent
    src_230_dead_code_attribute_parser_rs__detect_intent_signals --> src_400_dead_code_intent_rs__planned_directory_intent
    src_230_dead_code_attribute_parser_rs__detect_intent_signals --> src_400_dead_code_intent_rs__merge_intent_sources
    src_230_dead_code_attribute_parser_rs__detect_test_modules --> src_030_is_cfg_test_item_rs__is_cfg_test_item
    src_230_dead_code_attribute_parser_rs__detect_test_symbols --> src_410_dead_code_test_boundaries_rs__has_test_attr
    src_230_dead_code_attribute_parser_rs__detect_test_symbols --> src_030_is_cfg_test_item_rs__is_cfg_test_item
    src_280_file_ordering_rs__parallel_build_file_dag --> src_080_cluster_011_rs__build_directory_dag
    src_330_markdown_report_rs__collect_directory_files --> src_330_markdown_report_rs__collect_directory_files
    src_330_markdown_report_rs__generate_canonical_name --> src_120_cluster_006_rs__strip_numeric_prefix
    src_330_markdown_report_rs__collect_directory_moves --> src_010_layer_utilities_rs__compare_dir_layers
    src_330_markdown_report_rs__collect_directory_moves --> src_120_cluster_006_rs__strip_numeric_prefix
    src_330_markdown_report_rs__write_structural_batches --> src_330_markdown_report_rs__compress_path
    src_330_markdown_report_rs__write_structural_batches --> src_330_markdown_report_rs__compress_path
    src_330_markdown_report_rs__write_cluster_batches --> src_330_markdown_report_rs__compress_path
    src_330_markdown_report_rs__resolve_required_layer_path --> src_330_markdown_report_rs__collect_directory_files
    src_330_markdown_report_rs__resolve_required_layer_path --> src_330_markdown_report_rs__path_common_prefix_len
    src_330_markdown_report_rs__collect_move_items --> src_330_markdown_report_rs__compute_move_metrics
    src_330_markdown_report_rs__collect_move_items --> src_330_markdown_report_rs__compress_path
    src_330_markdown_report_rs__collect_move_items --> src_330_markdown_report_rs__resolve_required_layer_path
    src_330_markdown_report_rs__collect_move_items --> src_330_markdown_report_rs__compress_path
    src_330_markdown_report_rs__collect_move_items --> src_330_markdown_report_rs__compute_move_metrics
    src_340_main_rs__main --> src_350_agent_cli_rs__run_agent_cli
    src_340_main_rs__main --> src_170_layer_utilities_rs__main
    src_340_main_rs__main --> src_340_main_rs__main
    src_350_agent_cli_rs__run_agent_cli --> src_350_agent_cli_rs__check_action
    src_350_agent_cli_rs__run_agent_cli --> src_350_agent_cli_rs__query_function
    src_350_agent_cli_rs__run_agent_cli --> src_350_agent_cli_rs__list_invariants
    src_350_agent_cli_rs__run_agent_cli --> src_350_agent_cli_rs__show_stats
    src_350_agent_cli_rs__check_action --> src_350_agent_cli_rs__load_invariants
    src_350_agent_cli_rs__check_action --> src_350_agent_cli_rs__check_action
    src_350_agent_cli_rs__query_function --> src_350_agent_cli_rs__load_invariants
    src_350_agent_cli_rs__list_invariants --> src_350_agent_cli_rs__load_invariants
    src_350_agent_cli_rs__show_stats --> src_350_agent_cli_rs__load_invariants
    src_380_dead_code_doc_comment_parser_rs__extract_doc_markers --> src_380_dead_code_doc_comment_parser_rs__detect_latent_markers
    src_390_dead_code_call_graph_rs__is_reachable --> src_390_dead_code_call_graph_rs__compute_reachability
    src_390_dead_code_call_graph_rs__is_test_only --> src_390_dead_code_call_graph_rs__build_reverse_call_graph
    src_400_dead_code_intent_rs__planned_directory_intent --> src_400_dead_code_intent_rs__check_planned_directory
    src_400_dead_code_intent_rs__planned_directory_intent --> src_400_dead_code_intent_rs__collect_symbols
    src_410_dead_code_test_boundaries_rs__find_test_callers --> src_390_dead_code_call_graph_rs__build_reverse_call_graph
    src_420_dead_code_entrypoints_rs__collect_entrypoints --> src_420_dead_code_entrypoints_rs__treat_public_as_entrypoint
    src_420_dead_code_entrypoints_rs__collect_exports --> src_420_dead_code_entrypoints_rs__collect_use_tree_idents
    src_420_dead_code_entrypoints_rs__collect_use_tree_idents --> src_420_dead_code_entrypoints_rs__collect_use_tree_idents
    src_420_dead_code_entrypoints_rs__collect_use_tree_idents --> src_420_dead_code_entrypoints_rs__collect_use_tree_idents
    src_430_dead_code_classifier_rs__is_reachable --> src_390_dead_code_call_graph_rs__compute_reachability
    src_470_dead_code_report_rs__write_outputs --> src_470_dead_code_report_rs__write_report
    src_470_dead_code_report_rs__write_outputs --> src_540_dead_code_report_split_rs__write_summary_markdown
    src_470_dead_code_report_rs__write_outputs --> src_540_dead_code_report_split_rs__write_plan_markdown
    src_480_dead_code_filter_rs__filter_dead_code_elements --> src_480_dead_code_filter_rs__collect_excluded_symbols
    src_480_dead_code_filter_rs__collect_excluded_symbols --> src_480_dead_code_filter_rs__should_exclude_from_analysis
    src_520_dead_code_policy_rs__load_policy --> src_520_dead_code_policy_rs__parse_policy
    src_520_dead_code_policy_rs__parse_policy --> src_520_dead_code_policy_rs__parse_list
    src_520_dead_code_policy_rs__parse_policy --> src_520_dead_code_policy_rs__parse_list
    src_520_dead_code_policy_rs__parse_policy --> src_520_dead_code_policy_rs__parse_list
    src_520_dead_code_policy_rs__parse_policy --> src_520_dead_code_policy_rs__parse_bool
    src_540_dead_code_report_split_rs__write_summary_markdown --> src_540_dead_code_report_split_rs__top_items
    src_540_dead_code_report_split_rs__write_plan_markdown --> src_540_dead_code_report_split_rs__top_items
    src_540_dead_code_report_split_rs__write_plan_markdown --> src_540_dead_code_report_split_rs__plan_options
    src_620_correction_plan_serializer_rs__serialize_correction_plans --> src_620_correction_plan_serializer_rs__serialize_correction_plan
    src_620_correction_plan_serializer_rs__write_intelligence_outputs_at --> src_620_correction_plan_serializer_rs__serialize_correction_plans
    src_620_correction_plan_serializer_rs__write_intelligence_outputs_at --> src_630_verification_policy_emitter_rs__emit_verification_policy
    src_620_correction_plan_serializer_rs__write_intelligence_outputs --> src_620_correction_plan_serializer_rs__write_intelligence_outputs_at
    src_640_correction_intelligence_report_rs__filter_path_coherence_report --> src_640_correction_intelligence_report_rs__compute_summary
    src_640_correction_intelligence_report_rs__write_admission_preflight_report --> src_640_correction_intelligence_report_rs__generate_admission_preflight
    src_640_correction_intelligence_report_rs__generate_admission_preflight --> src_640_correction_intelligence_report_rs__evaluate_move_admission
    src_640_correction_intelligence_report_rs__evaluate_move_admission --> src_640_correction_intelligence_report_rs__find_function_definition_candidates
    src_640_correction_intelligence_report_rs__evaluate_move_admission --> src_640_correction_intelligence_report_rs__function_in_impl_block
    src_640_correction_intelligence_report_rs__evaluate_move_admission --> src_640_correction_intelligence_report_rs__is_test_scoped_function
    src_640_correction_intelligence_report_rs__evaluate_move_admission --> src_640_correction_intelligence_report_rs__find_private_dependencies
    src_640_correction_intelligence_report_rs__find_private_dependencies --> src_640_correction_intelligence_report_rs__extract_function_block_from_contents
    src_640_correction_intelligence_report_rs__find_private_dependencies --> src_250_cohesion_analyzer_rs__extract_identifiers
    src_640_correction_intelligence_report_rs__find_private_dependencies --> src_640_correction_intelligence_report_rs__extract_identifiers
    src_640_correction_intelligence_report_rs__extract_function_block_from_contents --> src_640_correction_intelligence_report_rs__is_function_signature_line
    src_640_correction_intelligence_report_rs__extract_identifiers --> src_640_correction_intelligence_report_rs__is_identifier_candidate
    src_640_correction_intelligence_report_rs__extract_identifiers --> src_640_correction_intelligence_report_rs__is_identifier_candidate
    src_640_correction_intelligence_report_rs__find_function_definition_candidates --> src_640_correction_intelligence_report_rs__function_signature_found
    src_640_correction_intelligence_report_rs__find_function_definition_candidates --> src_640_correction_intelligence_report_rs__function_in_impl_block
    src_640_correction_intelligence_report_rs__function_signature_found --> src_640_correction_intelligence_report_rs__is_function_signature_line
    src_640_correction_intelligence_report_rs__function_in_impl_block --> src_640_correction_intelligence_report_rs__is_function_signature_line
    src_640_correction_intelligence_report_rs__is_test_scoped_function --> src_640_correction_intelligence_report_rs__is_test_attribute_line
    src_640_correction_intelligence_report_rs__is_test_scoped_function --> src_640_correction_intelligence_report_rs__is_function_signature_line
    src_640_correction_intelligence_report_rs__filter_visibility_report --> src_640_correction_intelligence_report_rs__compute_summary
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice --> src_640_correction_intelligence_report_rs__parse_phase2_cluster_plan
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice --> src_640_correction_intelligence_report_rs__generate_correction_plan
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice --> src_640_correction_intelligence_report_rs__plan_verification_scope
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice --> src_640_correction_intelligence_report_rs__build_rollback_criteria
    src_640_correction_intelligence_report_rs__generate_phase2_cluster_slice --> src_640_correction_intelligence_report_rs__compute_summary
    src_640_correction_intelligence_report_rs__augment_path_coherence_strategies --> src_640_correction_intelligence_report_rs__module_name_from_path
    src_640_correction_intelligence_report_rs__augment_path_coherence_strategies --> src_640_correction_intelligence_report_rs__module_name_from_path
    src_640_correction_intelligence_report_rs__augment_path_coherence_strategies --> src_020_gather_rust_files_rs__gather_rust_files
    src_640_correction_intelligence_report_rs__module_name_from_path --> src_060_module_resolution_rs__normalize_module_name
    src_640_correction_intelligence_report_rs__fill_prediction_confidence --> src_640_correction_intelligence_report_rs__default_confidence
    src_640_correction_intelligence_report_rs__find_callers --> src_640_correction_intelligence_report_rs__find_element_file
    src_640_correction_intelligence_report_rs__find_reference_files --> src_640_correction_intelligence_report_rs__find_element_file
    src_640_correction_intelligence_report_rs__predict_violations --> src_640_correction_intelligence_report_rs__find_callers
    src_640_correction_intelligence_report_rs__predict_violations --> src_640_correction_intelligence_report_rs__move_violates_invariant
    src_640_correction_intelligence_report_rs__predict_violations --> src_640_correction_intelligence_report_rs__symbol_exists
    src_640_correction_intelligence_report_rs__predict_violations --> src_640_correction_intelligence_report_rs__find_reference_files
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_symbol
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_module_path
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_refs
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_refs
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_symbol
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_target_layer
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_function
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_function
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_target_layer
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__action_visibility
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__average_confidence
    src_640_correction_intelligence_report_rs__generate_correction_plan --> src_640_correction_intelligence_report_rs__estimate_fix_time
    src_640_correction_intelligence_report_rs__plan_verification_scope --> src_640_correction_intelligence_report_rs__affected_files
    src_640_correction_intelligence_report_rs__plan_verification_scope --> src_640_correction_intelligence_report_rs__action_module
    src_640_correction_intelligence_report_rs__plan_verification_scope --> src_640_correction_intelligence_report_rs__estimate_verification_time
    src_640_correction_intelligence_report_rs__build_rollback_criteria --> src_640_correction_intelligence_report_rs__extract_critical_tests
    src_640_correction_intelligence_report_rs__estimate_impact --> src_640_correction_intelligence_report_rs__simulate_action
    src_640_correction_intelligence_report_rs__estimate_impact --> src_640_correction_intelligence_report_rs__calculate_quality_delta
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__predict_violations
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__fill_prediction_confidence
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__generate_correction_plan
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__augment_path_coherence_strategies
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__plan_verification_scope
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__build_rollback_criteria
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__estimate_impact
    src_640_correction_intelligence_report_rs__generate_intelligence_report --> src_640_correction_intelligence_report_rs__compute_summary
```
