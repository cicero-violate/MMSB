# Module Exports

## src/060_module_resolution.rs (060_module_resolution.rs)

Module `060_module_resolution`

- `crate :: cluster_001 :: order_julia_files_by_dependency`
- `moved_gather_rust_files :: gather_rust_files`

## src/100_dependency.rs (100_dependency.rs)

Module `100_dependency`

- `# [allow (unused_imports)] pub use crate :: cluster_001 :: { detect_layer , julia_entry_paths , order_rust_files_by_dependency }`
- `# [allow (unused_imports)] pub use crate :: cluster_010 :: { extract_julia_dependencies , extract_rust_dependencies }`
- `crate :: cluster_001 :: { analyze_file_ordering , naming_score_for_file }`
- `crate :: cluster_001 :: { build_directory_entry_map , collect_naming_warnings }`
- `crate :: cluster_010 :: order_julia_files_by_dependency`
- `crate :: cluster_011 :: build_module_map`
- `crate :: cluster_011 :: { build_directory_dag , build_file_dependency_graph }`

## src/120_cluster_006.rs (120_cluster_006.rs)

Module `120_cluster_006`

- `# [allow (unused_imports)] pub use crate :: report :: collect_directory_moves`
- `# [allow (unused_imports)] pub use crate :: report :: generate_canonical_name`

## src/140_layer_core.rs (140_layer_core.rs)

Module `140_layer_core`

- `# [allow (unused_imports)] pub use crate :: cluster_001 :: { layer_constrained_sort , topo_sort_within }`
- `# [allow (unused_imports)] pub use crate :: cluster_006 :: { layer_prefix_value , order_directories , collect_directory_moves , }`
- `# [allow (unused_imports)] pub use crate :: cluster_008 :: detect_layer_violations`
- `crate :: cluster_008 :: sort_structural_items`

## src/170_layer_utilities.rs (170_layer_utilities.rs)

Module `170_layer_utilities`

- `# [allow (unused_imports)] pub use crate :: cluster_001 :: { build_file_layers , detect_layer , gather_julia_files , julia_entry_paths }`
- `# [allow (unused_imports)] pub use crate :: cluster_010 :: contains_tools`

## src/220_utilities.rs (220_utilities.rs)

Module `220_utilities`

- `# [allow (unused_imports)] pub use crate :: report :: collect_directory_files`
- `# [allow (unused_imports)] pub use crate :: report :: collect_move_items`
- `# [allow (unused_imports)] pub use crate :: report :: compute_move_metrics`
- `# [allow (unused_imports)] pub use crate :: report :: path_common_prefix_len`
- `# [allow (unused_imports)] pub use crate :: report :: resolve_required_layer_path`
- `# [allow (unused_imports)] pub use crate :: report :: write_cluster_batches`
- `# [allow (unused_imports)] pub use crate :: report :: write_structural_batches`
- `# [doc = " Compress absolute paths to MMSB-relative format"] # [allow (unused_imports)] pub use crate :: report :: compress_path`

## src/230_dead_code_attribute_parser.rs (230_dead_code_attribute_parser.rs)

Module `230_dead_code_attribute_parser`

- `moved_is_cfg_test_item :: is_cfg_test_item`
- `moved_scan_doc_comments :: scan_doc_comments`

## src/280_file_ordering.rs (280_file_ordering.rs)

Module `280_file_ordering`

- `crate :: cluster_001 :: { ordered_by_name , topological_sort }`
- `crate :: cluster_010 :: build_dependency_map`

## src/310_dot_exporter.rs (310_dot_exporter.rs)

Module `310_dot_exporter`

- `# [allow (unused_imports)] pub use crate :: cluster_001 :: export_complete_program_dot`
- `crate :: cluster_011 :: export_program_cfg_to_path`

## src/360_lib.rs (360_lib.rs)

Module `360_lib`

- `action_validator :: { AgentAction , ConstraintViolation , ViolationSeverity }`
- `admission_composition_artifact :: { generate_artifact , write_artifact , read_artifact , AdmissionCompositionArtifact , ARTIFACT_SCHEMA_VERSION , }`
- `agent_conscience :: { ActionPermission , AgentConscience }`
- `batch_admission :: { admit_batch , AdmissionDecision }`
- `cohesion_analyzer :: FunctionCohesionAnalyzer`
- `composition_rule :: { compose_batch , CompositionResult , ComposedEffectState , ConflictReason , InvariantType , }`
- `conscience_graph :: { generate_conscience_map , generate_conscience_stats }`
- `control_flow :: ControlFlowAnalyzer`
- `dependency :: { julia_entry_paths , order_julia_files_by_dependency , order_rust_files_by_dependency , LayerGraph , build_file_dependency_graph , analyze_file_ordering , }`
- `directory_analyzer :: DirectoryAnalyzer`
- `dot_exporter :: { export_complete_program_dot , export_program_cfg_to_path }`
- `effect_signature_schema :: { EffectSignature , ReadEffects , WriteEffects , StructuralTransitions , InvariantTouchpoints , ExecutorSurface , SCHEMA_VERSION , }`
- `file_ordering :: { DagCache , parallel_build_file_dag }`
- `invariant_types :: *`
- `julia_parser :: JuliaAnalyzer`
- `refactor_constraints :: *`
- `report :: ReportGenerator`
- `rust_parser :: RustAnalyzer`
- `types :: AnalysisResult`

## src/390_dead_code_call_graph.rs (390_dead_code_call_graph.rs)

Module `390_dead_code_call_graph`

- `moved_classify_symbol :: classify_symbol`

## src/400_dead_code_intent.rs (400_dead_code_intent.rs)

Module `400_dead_code_intent`

- `crate :: dead_code_attribute_parser :: detect_intent_signals`

## src/500_dead_code_cli.rs (500_dead_code_cli.rs)

Module `500_dead_code_cli`

- `crate :: layer_utilities :: run_dead_code_pipeline`

## src/530_violation_predictor.rs (530_violation_predictor.rs)

Module `530_violation_predictor`

- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: generate_intelligence_report`
- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: predict_violations`

## src/570_correction_plan_generator.rs (570_correction_plan_generator.rs)

Module `570_correction_plan_generator`

- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: generate_correction_plan`

## src/580_verification_scope_planner.rs (580_verification_scope_planner.rs)

Module `580_verification_scope_planner`

- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: plan_verification_scope`

## src/590_rollback_criteria_builder.rs (590_rollback_criteria_builder.rs)

Module `590_rollback_criteria_builder`

- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: build_rollback_criteria`

## src/600_quality_delta_calculator.rs (600_quality_delta_calculator.rs)

Module `600_quality_delta_calculator`

- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: calculate_quality_delta`
- `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: estimate_impact`

## src/610_action_impact_estimator.rs (610_action_impact_estimator.rs)

Module `610_action_impact_estimator`

- `# [allow (unused_imports)] pub use crate :: quality_delta_calculator :: estimate_impact`

## src/640_correction_intelligence_report.rs (640_correction_intelligence_report.rs)

Module `640_correction_intelligence_report`

- `# [allow (unused_imports)] pub use crate :: correction_plan_serializer :: write_intelligence_outputs`
- `# [allow (unused_imports)] pub use crate :: correction_plan_serializer :: write_intelligence_outputs_at`

