# MMSB Code Structure Overview

Generated: 2026-01-01 02:49:49

Each numbered file groups source files by MMSB prefix so a simple `ls 10_structure/` shows the traversal order.

## Group Files

- `src/000_dependency_analysis.rs` → `000-src-000_dependency_analysis_rs.md`
- `src/010_layer_utilities.rs` → `010-src-010_layer_utilities_rs.md`
- `src/020_gather_rust_files.rs` → `020-src-020_gather_rust_files_rs.md`
- `src/030_invariant_types.rs` → `030-src-030_invariant_types_rs.md`
- `src/030_is_cfg_test_item.rs` → `040-src-030_is_cfg_test_item_rs.md`
- `src/040_classify_symbol.rs` → `050-src-040_classify_symbol_rs.md`
- `src/040_refactor_constraints.rs` → `060-src-040_refactor_constraints_rs.md`
- `src/050_scc_compressor.rs` → `070-src-050_scc_compressor_rs.md`
- `src/060_module_resolution.rs` → `080-src-060_module_resolution_rs.md`
- `src/070_layer_inference.rs` → `090-src-070_layer_inference_rs.md`
- `src/080_cluster_011.rs` → `100-src-080_cluster_011_rs.md`
- `src/090_fixpoint_solver.rs` → `110-src-090_fixpoint_solver_rs.md`
- `src/100_dependency.rs` → `120-src-100_dependency_rs.md`
- `src/110_structural_detector.rs` → `130-src-110_structural_detector_rs.md`
- `src/120_cluster_006.rs` → `140-src-120_cluster_006_rs.md`
- `src/130_semantic_detector.rs` → `150-src-130_semantic_detector_rs.md`
- `src/150_path_detector.rs` → `160-src-150_path_detector_rs.md`
- `src/160_invariant_integrator.rs` → `170-src-160_invariant_integrator_rs.md`
- `src/170_layer_utilities.rs` → `180-src-170_layer_utilities_rs.md`
- `src/180_invariant_reporter.rs` → `190-src-180_invariant_reporter_rs.md`
- `src/190_conscience_graph.rs` → `200-src-190_conscience_graph_rs.md`
- `src/200_action_validator.rs` → `210-src-200_action_validator_rs.md`
- `src/210_agent_conscience.rs` → `220-src-210_agent_conscience_rs.md`
- `src/211_dead_code_doc_comment_scanner.rs` → `230-src-211_dead_code_doc_comment_scanner_rs.md`
- `src/230_dead_code_attribute_parser.rs` → `240-src-230_dead_code_attribute_parser_rs.md`
- `src/240_types.rs` → `250-src-240_types_rs.md`
- `src/250_cohesion_analyzer.rs` → `260-src-250_cohesion_analyzer_rs.md`
- `src/260_directory_analyzer.rs` → `270-src-260_directory_analyzer_rs.md`
- `src/270_control_flow.rs` → `280-src-270_control_flow_rs.md`
- `src/280_file_ordering.rs` → `290-src-280_file_ordering_rs.md`
- `src/290_julia_parser.rs` → `300-src-290_julia_parser_rs.md`
- `src/300_rust_parser.rs` → `310-src-300_rust_parser_rs.md`
- `src/330_markdown_report.rs` → `320-src-330_markdown_report_rs.md`
- `src/340_main.rs` → `330-src-340_main_rs.md`
- `src/350_agent_cli.rs` → `340-src-350_agent_cli_rs.md`
- `src/360_lib.rs` → `350-src-360_lib_rs.md`
- `src/370_dead_code_types.rs` → `360-src-370_dead_code_types_rs.md`
- `src/380_dead_code_doc_comment_parser.rs` → `370-src-380_dead_code_doc_comment_parser_rs.md`
- `src/390_dead_code_call_graph.rs` → `380-src-390_dead_code_call_graph_rs.md`
- `src/400_dead_code_intent.rs` → `390-src-400_dead_code_intent_rs.md`
- `src/410_dead_code_test_boundaries.rs` → `400-src-410_dead_code_test_boundaries_rs.md`
- `src/420_dead_code_entrypoints.rs` → `410-src-420_dead_code_entrypoints_rs.md`
- `src/430_dead_code_classifier.rs` → `420-src-430_dead_code_classifier_rs.md`
- `src/440_dead_code_confidence.rs` → `430-src-440_dead_code_confidence_rs.md`
- `src/450_dead_code_actions.rs` → `440-src-450_dead_code_actions_rs.md`
- `src/460_correction_plan_types.rs` → `450-src-460_correction_plan_types_rs.md`
- `src/470_dead_code_report.rs` → `460-src-470_dead_code_report_rs.md`
- `src/480_dead_code_filter.rs` → `470-src-480_dead_code_filter_rs.md`
- `src/490_verification_policy_types.rs` → `480-src-490_verification_policy_types_rs.md`
- `src/500_dead_code_cli.rs` → `490-src-500_dead_code_cli_rs.md`
- `src/510_quality_delta_types.rs` → `500-src-510_quality_delta_types_rs.md`
- `src/520_dead_code_policy.rs` → `510-src-520_dead_code_policy_rs.md`
- `src/540_dead_code_report_split.rs` → `520-src-540_dead_code_report_split_rs.md`
- `src/550_tier_classifier.rs` → `530-src-550_tier_classifier_rs.md`
- `src/560_confidence_scorer.rs` → `540-src-560_confidence_scorer_rs.md`
- `src/600_quality_delta_calculator.rs` → `550-src-600_quality_delta_calculator_rs.md`
- `src/610_action_impact_estimator.rs` → `560-src-610_action_impact_estimator_rs.md`
- `src/620_correction_plan_serializer.rs` → `570-src-620_correction_plan_serializer_rs.md`
- `src/630_verification_policy_emitter.rs` → `580-src-630_verification_policy_emitter_rs.md`
- `src/640_correction_intelligence_report.rs` → `590-src-640_correction_intelligence_report_rs.md`
- `src/admission_composition_artifact.rs` → `600-src-admission_composition_artifact_rs.md`
- `src/batch_admission.rs` → `610-src-batch_admission_rs.md`
- `src/composition_rule.rs` → `620-src-composition_rule_rs.md`
- `src/effect_signature_schema.rs` → `630-src-effect_signature_schema_rs.md`

## Summary Statistics

- Total elements: 577
- Rust elements: 577
- Julia elements: 0

### Elements by Type

- Rust_Enum: 45
- Rust_Function: 214
- Rust_Impl: 44
- Rust_Module: 150
- Rust_Struct: 124
