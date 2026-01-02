# File Ordering Report

Generated: 2026-01-01 02:49:49

## Rust File Ordering

### Metrics

- Total files: 72
- Rename suggestions: 68
- Ordering violations: 0
- Layer violations: 114
- Directories: 1

### Cycles Detected
- MMSB/tools/mmsb-analyzer/src/540_dead_code_report_split.rs, MMSB/tools/mmsb-analyzer/src/470_dead_code_report.rs, MMSB/tools/mmsb-analyzer/src/500_dead_code_cli.rs, MMSB/tools/mmsb-analyzer/src/170_layer_utilities.rs
- MMSB/tools/mmsb-analyzer/src/610_action_impact_estimator.rs, MMSB/tools/mmsb-analyzer/src/600_quality_delta_calculator.rs, MMSB/tools/mmsb-analyzer/src/620_correction_plan_serializer.rs, MMSB/tools/mmsb-analyzer/src/640_correction_intelligence_report.rs
- MMSB/tools/mmsb-analyzer/src/400_dead_code_intent.rs, MMSB/tools/mmsb-analyzer/src/230_dead_code_attribute_parser.rs
- MMSB/tools/mmsb-analyzer/src/410_dead_code_test_boundaries.rs, MMSB/tools/mmsb-analyzer/src/390_dead_code_call_graph.rs

### Canonical Order

| Order | Current | Suggested | Rename |
| --- | --- | --- | --- |
| 0 | `src/000_dependency_analysis.rs` | `000_dependency_analysis.rs` | no |
| 10 | `src/010_layer_utilities.rs` | `010_layer_utilities.rs` | no |
| 20 | `src/020_gather_rust_files.rs` | `020_gather_rust_files.rs` | no |
| 30 | `src/030_invariant_types.rs` | `030_invariant_types.rs` | no |
| 40 | `src/030_is_cfg_test_item.rs` | `040_is_cfg_test_item.rs` | yes |
| 50 | `src/040_classify_symbol.rs` | `050_classify_symbol.rs` | yes |
| 60 | `src/040_refactor_constraints.rs` | `060_refactor_constraints.rs` | yes |
| 70 | `src/050_scc_compressor.rs` | `070_scc_compressor.rs` | yes |
| 80 | `src/060_module_resolution.rs` | `080_module_resolution.rs` | yes |
| 90 | `src/070_layer_inference.rs` | `090_layer_inference.rs` | yes |
| 100 | `src/080_cluster_011.rs` | `100_cluster_011.rs` | yes |
| 110 | `src/090_fixpoint_solver.rs` | `110_fixpoint_solver.rs` | yes |
| 120 | `src/100_dependency.rs` | `120_dependency.rs` | yes |
| 130 | `src/110_structural_detector.rs` | `130_structural_detector.rs` | yes |
| 140 | `src/120_cluster_006.rs` | `140_cluster_006.rs` | yes |
| 150 | `src/130_semantic_detector.rs` | `150_semantic_detector.rs` | yes |
| 160 | `src/140_layer_core.rs` | `160_layer_core.rs` | yes |
| 170 | `src/150_path_detector.rs` | `170_path_detector.rs` | yes |
| 180 | `src/160_invariant_integrator.rs` | `180_invariant_integrator.rs` | yes |
| 190 | `src/170_layer_utilities.rs` | `190_layer_utilities.rs` | yes |
| 200 | `src/180_invariant_reporter.rs` | `200_invariant_reporter.rs` | yes |
| 210 | `src/190_conscience_graph.rs` | `210_conscience_graph.rs` | yes |
| 220 | `src/200_action_validator.rs` | `220_action_validator.rs` | yes |
| 230 | `src/210_agent_conscience.rs` | `230_agent_conscience.rs` | yes |
| 240 | `src/211_dead_code_doc_comment_scanner.rs` | `240_dead_code_doc_comment_scanner.rs` | yes |
| 250 | `src/220_utilities.rs` | `250_utilities.rs` | yes |
| 260 | `src/230_dead_code_attribute_parser.rs` | `260_dead_code_attribute_parser.rs` | yes |
| 270 | `src/240_types.rs` | `270_types.rs` | yes |
| 280 | `src/250_cohesion_analyzer.rs` | `280_cohesion_analyzer.rs` | yes |
| 290 | `src/260_directory_analyzer.rs` | `290_directory_analyzer.rs` | yes |
| 300 | `src/270_control_flow.rs` | `300_control_flow.rs` | yes |
| 310 | `src/280_file_ordering.rs` | `310_file_ordering.rs` | yes |
| 320 | `src/290_julia_parser.rs` | `320_julia_parser.rs` | yes |
| 330 | `src/300_rust_parser.rs` | `330_rust_parser.rs` | yes |
| 340 | `src/310_dot_exporter.rs` | `340_dot_exporter.rs` | yes |
| 350 | `src/320_file_gathering.rs` | `350_file_gathering.rs` | yes |
| 360 | `src/330_markdown_report.rs` | `360_markdown_report.rs` | yes |
| 370 | `src/340_main.rs` | `370_main.rs` | yes |
| 380 | `src/350_agent_cli.rs` | `380_agent_cli.rs` | yes |
| 390 | `src/360_lib.rs` | `390_lib.rs` | yes |
| 400 | `src/370_dead_code_types.rs` | `400_dead_code_types.rs` | yes |
| 410 | `src/380_dead_code_doc_comment_parser.rs` | `410_dead_code_doc_comment_parser.rs` | yes |
| 420 | `src/390_dead_code_call_graph.rs` | `420_dead_code_call_graph.rs` | yes |
| 430 | `src/400_dead_code_intent.rs` | `430_dead_code_intent.rs` | yes |
| 440 | `src/410_dead_code_test_boundaries.rs` | `440_dead_code_test_boundaries.rs` | yes |
| 450 | `src/420_dead_code_entrypoints.rs` | `450_dead_code_entrypoints.rs` | yes |
| 460 | `src/430_dead_code_classifier.rs` | `460_dead_code_classifier.rs` | yes |
| 470 | `src/440_dead_code_confidence.rs` | `470_dead_code_confidence.rs` | yes |
| 480 | `src/450_dead_code_actions.rs` | `480_dead_code_actions.rs` | yes |
| 490 | `src/460_correction_plan_types.rs` | `490_correction_plan_types.rs` | yes |
| 500 | `src/470_dead_code_report.rs` | `500_dead_code_report.rs` | yes |
| 510 | `src/480_dead_code_filter.rs` | `510_dead_code_filter.rs` | yes |
| 520 | `src/490_verification_policy_types.rs` | `520_verification_policy_types.rs` | yes |
| 530 | `src/500_dead_code_cli.rs` | `530_dead_code_cli.rs` | yes |
| 540 | `src/510_quality_delta_types.rs` | `540_quality_delta_types.rs` | yes |
| 550 | `src/520_dead_code_policy.rs` | `550_dead_code_policy.rs` | yes |
| 560 | `src/530_violation_predictor.rs` | `560_violation_predictor.rs` | yes |
| 570 | `src/540_dead_code_report_split.rs` | `570_dead_code_report_split.rs` | yes |
| 580 | `src/550_tier_classifier.rs` | `580_tier_classifier.rs` | yes |
| 590 | `src/560_confidence_scorer.rs` | `590_confidence_scorer.rs` | yes |
| 600 | `src/570_correction_plan_generator.rs` | `600_correction_plan_generator.rs` | yes |
| 610 | `src/580_verification_scope_planner.rs` | `610_verification_scope_planner.rs` | yes |
| 620 | `src/590_rollback_criteria_builder.rs` | `620_rollback_criteria_builder.rs` | yes |
| 630 | `src/600_quality_delta_calculator.rs` | `630_quality_delta_calculator.rs` | yes |
| 640 | `src/610_action_impact_estimator.rs` | `640_action_impact_estimator.rs` | yes |
| 650 | `src/620_correction_plan_serializer.rs` | `650_correction_plan_serializer.rs` | yes |
| 660 | `src/630_verification_policy_emitter.rs` | `660_verification_policy_emitter.rs` | yes |
| 670 | `src/640_correction_intelligence_report.rs` | `670_correction_intelligence_report.rs` | yes |
| 680 | `src/admission_composition_artifact.rs` | `680_admission_composition_artifact.rs` | yes |
| 690 | `src/batch_admission.rs` | `690_batch_admission.rs` | yes |
| 700 | `src/composition_rule.rs` | `700_composition_rule.rs` | yes |
| 710 | `src/effect_signature_schema.rs` | `710_effect_signature_schema.rs` | yes |

### Ordering Violations
- None detected.

### Layer Violations
- `src/010_layer_utilities.rs` (010_layer_utilities.rs) depends on `src/100_dependency.rs` (100_dependency.rs)
- `src/010_layer_utilities.rs` (010_layer_utilities.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/450_dead_code_actions.rs` (450_dead_code_actions.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/230_dead_code_attribute_parser.rs` (230_dead_code_attribute_parser.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/390_dead_code_call_graph.rs` (390_dead_code_call_graph.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/500_dead_code_cli.rs` (500_dead_code_cli.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/440_dead_code_confidence.rs` (440_dead_code_confidence.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/420_dead_code_entrypoints.rs` (420_dead_code_entrypoints.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/400_dead_code_intent.rs` (400_dead_code_intent.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/470_dead_code_report.rs` (470_dead_code_report.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/170_layer_utilities.rs` (170_layer_utilities.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/570_correction_plan_generator.rs` (570_correction_plan_generator.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/020_gather_rust_files.rs` (020_gather_rust_files.rs) depends on `src/170_layer_utilities.rs` (170_layer_utilities.rs)
- `src/230_dead_code_attribute_parser.rs` (230_dead_code_attribute_parser.rs) depends on `src/380_dead_code_doc_comment_parser.rs` (380_dead_code_doc_comment_parser.rs)
- `src/230_dead_code_attribute_parser.rs` (230_dead_code_attribute_parser.rs) depends on `src/400_dead_code_intent.rs` (400_dead_code_intent.rs)
- `src/230_dead_code_attribute_parser.rs` (230_dead_code_attribute_parser.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/230_dead_code_attribute_parser.rs` (230_dead_code_attribute_parser.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/530_violation_predictor.rs` (530_violation_predictor.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/030_is_cfg_test_item.rs` (030_is_cfg_test_item.rs) depends on `src/380_dead_code_doc_comment_parser.rs` (380_dead_code_doc_comment_parser.rs)
- `src/110_structural_detector.rs` (110_structural_detector.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/610_action_impact_estimator.rs` (610_action_impact_estimator.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/350_agent_cli.rs` (350_agent_cli.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/560_confidence_scorer.rs` (560_confidence_scorer.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/570_correction_plan_generator.rs` (570_correction_plan_generator.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/620_correction_plan_serializer.rs` (620_correction_plan_serializer.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/460_correction_plan_types.rs` (460_correction_plan_types.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/450_dead_code_actions.rs` (450_dead_code_actions.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/390_dead_code_call_graph.rs` (390_dead_code_call_graph.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/430_dead_code_classifier.rs` (430_dead_code_classifier.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/500_dead_code_cli.rs` (500_dead_code_cli.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/440_dead_code_confidence.rs` (440_dead_code_confidence.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/380_dead_code_doc_comment_parser.rs` (380_dead_code_doc_comment_parser.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/420_dead_code_entrypoints.rs` (420_dead_code_entrypoints.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/480_dead_code_filter.rs` (480_dead_code_filter.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/400_dead_code_intent.rs` (400_dead_code_intent.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/520_dead_code_policy.rs` (520_dead_code_policy.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/470_dead_code_report.rs` (470_dead_code_report.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/540_dead_code_report_split.rs` (540_dead_code_report_split.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/600_quality_delta_calculator.rs` (600_quality_delta_calculator.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/510_quality_delta_types.rs` (510_quality_delta_types.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/590_rollback_criteria_builder.rs` (590_rollback_criteria_builder.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/550_tier_classifier.rs` (550_tier_classifier.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/630_verification_policy_emitter.rs` (630_verification_policy_emitter.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/490_verification_policy_types.rs` (490_verification_policy_types.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/580_verification_scope_planner.rs` (580_verification_scope_planner.rs)
- `src/340_main.rs` (340_main.rs) depends on `src/530_violation_predictor.rs` (530_violation_predictor.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/610_action_impact_estimator.rs` (610_action_impact_estimator.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/560_confidence_scorer.rs` (560_confidence_scorer.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/570_correction_plan_generator.rs` (570_correction_plan_generator.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/620_correction_plan_serializer.rs` (620_correction_plan_serializer.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/460_correction_plan_types.rs` (460_correction_plan_types.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/450_dead_code_actions.rs` (450_dead_code_actions.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/390_dead_code_call_graph.rs` (390_dead_code_call_graph.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/430_dead_code_classifier.rs` (430_dead_code_classifier.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/500_dead_code_cli.rs` (500_dead_code_cli.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/440_dead_code_confidence.rs` (440_dead_code_confidence.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/380_dead_code_doc_comment_parser.rs` (380_dead_code_doc_comment_parser.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/420_dead_code_entrypoints.rs` (420_dead_code_entrypoints.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/480_dead_code_filter.rs` (480_dead_code_filter.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/400_dead_code_intent.rs` (400_dead_code_intent.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/520_dead_code_policy.rs` (520_dead_code_policy.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/470_dead_code_report.rs` (470_dead_code_report.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/540_dead_code_report_split.rs` (540_dead_code_report_split.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/600_quality_delta_calculator.rs` (600_quality_delta_calculator.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/510_quality_delta_types.rs` (510_quality_delta_types.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/590_rollback_criteria_builder.rs` (590_rollback_criteria_builder.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/550_tier_classifier.rs` (550_tier_classifier.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/630_verification_policy_emitter.rs` (630_verification_policy_emitter.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/490_verification_policy_types.rs` (490_verification_policy_types.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/580_verification_scope_planner.rs` (580_verification_scope_planner.rs)
- `src/360_lib.rs` (360_lib.rs) depends on `src/530_violation_predictor.rs` (530_violation_predictor.rs)
- `src/130_semantic_detector.rs` (130_semantic_detector.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/040_classify_symbol.rs` (040_classify_symbol.rs) depends on `src/390_dead_code_call_graph.rs` (390_dead_code_call_graph.rs)
- `src/040_classify_symbol.rs` (040_classify_symbol.rs) depends on `src/400_dead_code_intent.rs` (400_dead_code_intent.rs)
- `src/040_classify_symbol.rs` (040_classify_symbol.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/040_classify_symbol.rs` (040_classify_symbol.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/470_dead_code_report.rs` (470_dead_code_report.rs) depends on `src/500_dead_code_cli.rs` (500_dead_code_cli.rs)
- `src/470_dead_code_report.rs` (470_dead_code_report.rs) depends on `src/540_dead_code_report_split.rs` (540_dead_code_report_split.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/120_cluster_006.rs` (120_cluster_006.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/250_cohesion_analyzer.rs` (250_cohesion_analyzer.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/270_control_flow.rs` (270_control_flow.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/100_dependency.rs` (100_dependency.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/260_directory_analyzer.rs` (260_directory_analyzer.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/310_dot_exporter.rs` (310_dot_exporter.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/280_file_ordering.rs` (280_file_ordering.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/160_invariant_integrator.rs` (160_invariant_integrator.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/180_invariant_reporter.rs` (180_invariant_reporter.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/290_julia_parser.rs` (290_julia_parser.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/140_layer_core.rs` (140_layer_core.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/170_layer_utilities.rs` (170_layer_utilities.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/300_rust_parser.rs` (300_rust_parser.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/000_dependency_analysis.rs` (000_dependency_analysis.rs) depends on `src/220_utilities.rs` (220_utilities.rs)
- `src/160_invariant_integrator.rs` (160_invariant_integrator.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/620_correction_plan_serializer.rs` (620_correction_plan_serializer.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/620_correction_plan_serializer.rs` (620_correction_plan_serializer.rs) depends on `src/630_verification_policy_emitter.rs` (630_verification_policy_emitter.rs)
- `src/610_action_impact_estimator.rs` (610_action_impact_estimator.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/060_module_resolution.rs` (060_module_resolution.rs) depends on `src/100_dependency.rs` (100_dependency.rs)
- `src/060_module_resolution.rs` (060_module_resolution.rs) depends on `src/170_layer_utilities.rs` (170_layer_utilities.rs)
- `src/390_dead_code_call_graph.rs` (390_dead_code_call_graph.rs) depends on `src/410_dead_code_test_boundaries.rs` (410_dead_code_test_boundaries.rs)
- `src/600_quality_delta_calculator.rs` (600_quality_delta_calculator.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/080_cluster_011.rs` (080_cluster_011.rs) depends on `src/240_types.rs` (240_types.rs)
- `src/211_dead_code_doc_comment_scanner.rs` (211_dead_code_doc_comment_scanner.rs) depends on `src/380_dead_code_doc_comment_parser.rs` (380_dead_code_doc_comment_parser.rs)
- `src/211_dead_code_doc_comment_scanner.rs` (211_dead_code_doc_comment_scanner.rs) depends on `src/370_dead_code_types.rs` (370_dead_code_types.rs)
- `src/590_rollback_criteria_builder.rs` (590_rollback_criteria_builder.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)
- `src/580_verification_scope_planner.rs` (580_verification_scope_planner.rs) depends on `src/640_correction_intelligence_report.rs` (640_correction_intelligence_report.rs)

### Directory Order
- `src`

## Julia File Ordering

No files analyzed.

