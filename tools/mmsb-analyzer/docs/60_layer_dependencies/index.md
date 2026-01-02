# Layer Dependency Report

Generated: 2026-01-01 02:49:49

## Rust Layer Graph

### Layer Order
1. `root`
2. `000_dependency_analysis.rs`
3. `010_layer_utilities.rs`
4. `020_gather_rust_files.rs`
5. `030_invariant_types.rs`
6. `030_is_cfg_test_item.rs`
7. `040_classify_symbol.rs`
8. `040_refactor_constraints.rs`
9. `050_scc_compressor.rs`
10. `060_module_resolution.rs`
11. `070_layer_inference.rs`
12. `080_cluster_011.rs`
13. `090_fixpoint_solver.rs`
14. `100_dependency.rs`
15. `110_structural_detector.rs`
16. `120_cluster_006.rs`
17. `130_semantic_detector.rs`
18. `140_layer_core.rs`
19. `150_path_detector.rs`
20. `160_invariant_integrator.rs`
21. `170_layer_utilities.rs`
22. `180_invariant_reporter.rs`
23. `190_conscience_graph.rs`
24. `200_action_validator.rs`
25. `210_agent_conscience.rs`
26. `211_dead_code_doc_comment_scanner.rs`
27. `220_utilities.rs`
28. `230_dead_code_attribute_parser.rs`
29. `240_types.rs`
30. `250_cohesion_analyzer.rs`
31. `260_directory_analyzer.rs`
32. `270_control_flow.rs`
33. `280_file_ordering.rs`
34. `290_julia_parser.rs`
35. `300_rust_parser.rs`
36. `310_dot_exporter.rs`
37. `320_file_gathering.rs`
38. `330_markdown_report.rs`
39. `340_main.rs`
40. `350_agent_cli.rs`
41. `360_lib.rs`
42. `370_dead_code_types.rs`
43. `380_dead_code_doc_comment_parser.rs`
44. `390_dead_code_call_graph.rs`
45. `400_dead_code_intent.rs`
46. `410_dead_code_test_boundaries.rs`
47. `420_dead_code_entrypoints.rs`
48. `430_dead_code_classifier.rs`
49. `440_dead_code_confidence.rs`
50. `450_dead_code_actions.rs`
51. `460_correction_plan_types.rs`
52. `470_dead_code_report.rs`
53. `480_dead_code_filter.rs`
54. `490_verification_policy_types.rs`
55. `500_dead_code_cli.rs`
56. `510_quality_delta_types.rs`
57. `520_dead_code_policy.rs`
58. `530_violation_predictor.rs`
59. `540_dead_code_report_split.rs`
60. `550_tier_classifier.rs`
61. `560_confidence_scorer.rs`
62. `570_correction_plan_generator.rs`
63. `580_verification_scope_planner.rs`
64. `590_rollback_criteria_builder.rs`
65. `600_quality_delta_calculator.rs`
66. `610_action_impact_estimator.rs`
67. `620_correction_plan_serializer.rs`
68. `630_verification_policy_emitter.rs`
69. `640_correction_intelligence_report.rs`

### Layer Violations
- None detected.

### Dependency Edges
- No cross-layer dependencies recorded.

### Unresolved References
- src/250_cohesion_analyzer.rs → `use crate :: cluster_006 :: compute_cohesion_score ;`
- src/250_cohesion_analyzer.rs → `use crate :: cluster_008 :: { detect_layer_violation , FunctionInfo } ;`
- src/250_cohesion_analyzer.rs → `use crate :: types :: { CallAnalysis , FunctionCluster , FunctionPlacement , PlacementStatus } ;`
- src/250_cohesion_analyzer.rs → `use crate :: types :: { AnalysisResult , ElementType } ;`
- src/100_dependency.rs → `pub use crate :: cluster_010 :: order_julia_files_by_dependency ;`
- src/100_dependency.rs → `# [allow (unused_imports)] pub use crate :: cluster_001 :: { detect_layer , julia_entry_paths , order_rust_files_by_dependency } ;`
- src/100_dependency.rs → `pub use crate :: cluster_011 :: build_module_map ;`
- src/100_dependency.rs → `pub use crate :: cluster_001 :: { build_directory_entry_map , collect_naming_warnings } ;`
- src/100_dependency.rs → `# [allow (unused_imports)] pub use crate :: cluster_010 :: { extract_julia_dependencies , extract_rust_dependencies } ;`
- src/100_dependency.rs → `pub use crate :: cluster_001 :: { analyze_file_ordering , naming_score_for_file } ;`
- src/100_dependency.rs → `pub use crate :: cluster_011 :: { build_directory_dag , build_file_dependency_graph } ;`
- src/210_agent_conscience.rs → `use crate :: action_validator :: { validate_action , AgentAction , ConstraintViolation } ;`
- src/210_agent_conscience.rs → `use crate :: invariant_types :: { Invariant , InvariantStrength } ;`
- src/210_agent_conscience.rs → `use crate :: refactor_constraints :: { from_invariant , RefactorConstraint } ;`
- src/210_agent_conscience.rs → `use crate :: invariant_types :: { InvariantKind , SemanticInvariant } ;`
- src/210_agent_conscience.rs → `use crate :: invariant_types :: { InvariantKind , StructuralInvariant } ;`
- src/composition_rule.rs → `use crate :: effect_signature_schema :: * ;`
- src/420_dead_code_entrypoints.rs → `use crate :: dead_code_intent :: DeadCodePolicy ;`
- src/420_dead_code_entrypoints.rs → `use crate :: types :: { CodeElement , ElementType , Visibility } ;`
- src/520_dead_code_policy.rs → `use crate :: dead_code_intent :: DeadCodePolicy ;`
- src/211_dead_code_doc_comment_scanner.rs → `use crate :: dead_code_doc_comment_parser :: extract_doc_markers ;`
- src/211_dead_code_doc_comment_scanner.rs → `use crate :: dead_code_doc_comment_parser :: item_attrs ;`
- src/211_dead_code_doc_comment_scanner.rs → `use crate :: dead_code_doc_comment_parser :: item_name ;`
- src/211_dead_code_doc_comment_scanner.rs → `use crate :: dead_code_types :: IntentMarker ;`
- src/540_dead_code_report_split.rs → `use crate :: dead_code_report :: DeadCodeReportWithMeta ;`
- src/540_dead_code_report_split.rs → `use crate :: dead_code_types :: { DeadCodeCategory , DeadCodeItem , RecommendedAction } ;`
- src/290_julia_parser.rs → `use crate :: types :: * ;`
- src/010_layer_utilities.rs → `use crate :: dependency :: { LayerEdge , LayerGraph , ReferenceDetail , UnresolvedDependency , } ;`
- src/010_layer_utilities.rs → `use crate :: types :: { FileLayerViolation , NodeType } ;`
- src/460_correction_plan_types.rs → `use crate :: types :: Visibility ;`
- src/640_correction_intelligence_report.rs → `use crate :: correction_plan_types :: { CorrectionPlan , CorrectionStrategy , ErrorTier , RefactorAction , Severity , ViolationPrediction , ViolationType , } ;`
- src/640_correction_intelligence_report.rs → `use crate :: quality_delta_calculator :: Metrics ;`
- src/640_correction_intelligence_report.rs → `use crate :: quality_delta_types :: { QualityDelta , RollbackCriteria } ;`
- src/640_correction_intelligence_report.rs → `use crate :: invariant_types :: InvariantAnalysisResult ;`
- src/640_correction_intelligence_report.rs → `use crate :: types :: { AnalysisResult , CallGraphNode , CodeElement } ;`
- src/640_correction_intelligence_report.rs → `use crate :: action_impact_estimator :: AnalysisState as ImpactState ;`
- src/640_correction_intelligence_report.rs → `# [allow (unused_imports)] pub use crate :: correction_plan_serializer :: write_intelligence_outputs ;`
- src/640_correction_intelligence_report.rs → `# [allow (unused_imports)] pub use crate :: correction_plan_serializer :: write_intelligence_outputs_at ;`
- src/640_correction_intelligence_report.rs → `use crate :: action_impact_estimator :: AnalysisState ;`
- src/640_correction_intelligence_report.rs → `use crate :: correction_plan_types :: VisibilityPlanOption ;`
- src/640_correction_intelligence_report.rs → `use crate :: tier_classifier :: classify_tier ;`
- src/640_correction_intelligence_report.rs → `use crate :: verification_policy_types :: VerificationCheck ;`
- src/640_correction_intelligence_report.rs → `use crate :: verification_policy_types :: VerificationPolicy ;`
- src/640_correction_intelligence_report.rs → `use crate :: verification_policy_types :: VerificationScope ;`
- src/640_correction_intelligence_report.rs → `use crate :: quality_delta_types :: RollbackCondition ;`
- src/030_is_cfg_test_item.rs → `use crate :: dead_code_doc_comment_parser :: item_attrs ;`
- src/560_confidence_scorer.rs → `use crate :: correction_plan_types :: { ViolationPrediction , ViolationType } ;`
- src/080_cluster_011.rs → `use crate :: types :: ProgramCFG ;`
- src/470_dead_code_report.rs → `use crate :: dead_code_cli :: DeadCodeRunConfig ;`
- src/470_dead_code_report.rs → `use crate :: dead_code_report_split :: { write_plan_markdown , write_summary_markdown } ;`
- src/470_dead_code_report.rs → `use crate :: dead_code_types :: { DeadCodeItem , DeadCodeReport , DeadCodeSummary } ;`
- src/480_dead_code_filter.rs → `use crate :: dead_code_types :: { DeadCodeCategory } ;`
- src/480_dead_code_filter.rs → `use crate :: dead_code_report :: DeadCodeReportWithMeta ;`
- src/480_dead_code_filter.rs → `use crate :: types :: CodeElement ;`
- src/200_action_validator.rs → `use crate :: refactor_constraints :: RefactorConstraint ;`
- src/200_action_validator.rs → `use crate :: invariant_types :: InvariantStrength ;`
- src/060_module_resolution.rs → `use crate :: layer_utilities :: resolve_source_root ;`
- src/060_module_resolution.rs → `pub use crate :: cluster_001 :: order_julia_files_by_dependency ;`
- src/060_module_resolution.rs → `use crate :: dependency :: RootState ;`
- src/270_control_flow.rs → `use crate :: types :: * ;`
- src/270_control_flow.rs → `use crate :: utilities :: compress_path ;`
- src/260_directory_analyzer.rs → `use crate :: dependency :: detect_layer ;`
- src/260_directory_analyzer.rs → `use crate :: layer_utilities :: allow_analysis_dir ;`
- src/260_directory_analyzer.rs → `use crate :: types :: DirectoryAnalysis ;`
- src/600_quality_delta_calculator.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: estimate_impact ;`
- src/600_quality_delta_calculator.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: calculate_quality_delta ;`
- src/530_violation_predictor.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: generate_intelligence_report ;`
- src/530_violation_predictor.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: predict_violations ;`
- src/330_markdown_report.rs → `use crate :: cluster_008 :: collect_cluster_plans ;`
- src/330_markdown_report.rs → `use crate :: layer_core :: { sort_structural_items } ;`
- src/330_markdown_report.rs → `use crate :: control_flow :: ControlFlowAnalyzer ;`
- src/330_markdown_report.rs → `use crate :: dependency :: { LayerGraph , build_directory_entry_map , build_file_dependency_graph , collect_naming_warnings } ;`
- src/330_markdown_report.rs → `use crate :: file_ordering :: DirectoryMove ;`
- src/330_markdown_report.rs → `use crate :: types :: { AnalysisResult , CallGraphNode , CodeElement , DirectoryAnalysis , ElementType , FileOrderingResult , FunctionCfg , FunctionCluster , FunctionPlacement , Language , PlacementStatus , Visibility , } ;`
- src/330_markdown_report.rs → `use crate :: cluster_006 :: strip_numeric_prefix ;`
- src/330_markdown_report.rs → `use crate :: types :: { ElementType , Language , Visibility } ;`
- src/330_markdown_report.rs → `use crate :: refactor_constraints :: RefactorConstraint ;`
- src/330_markdown_report.rs → `use crate :: action_validator :: check_move_allowed ;`
- src/550_tier_classifier.rs → `use crate :: correction_plan_types :: { ErrorTier , Severity , ViolationPrediction , ViolationType } ;`
- src/620_correction_plan_serializer.rs → `use crate :: correction_intelligence_report :: CorrectionIntelligenceReport ;`
- src/620_correction_plan_serializer.rs → `use crate :: correction_plan_types :: { CorrectionPlan , CorrectionStrategy } ;`
- src/620_correction_plan_serializer.rs → `use crate :: quality_delta_types :: RollbackCriteria ;`
- src/620_correction_plan_serializer.rs → `use crate :: verification_policy_emitter :: emit_verification_policy ;`
- src/620_correction_plan_serializer.rs → `use crate :: verification_policy_types :: { QualityThresholds , VerificationCheck , VerificationPolicy , VerificationScope } ;`
- src/120_cluster_006.rs → `use crate :: cluster_008 :: FunctionInfo ;`
- src/120_cluster_006.rs → `# [allow (unused_imports)] pub use crate :: report :: collect_directory_moves ;`
- src/120_cluster_006.rs → `# [allow (unused_imports)] pub use crate :: report :: generate_canonical_name ;`
- src/430_dead_code_classifier.rs → `use crate :: dead_code_call_graph :: CallGraph ;`
- src/350_agent_cli.rs → `use crate :: action_validator :: AgentAction ;`
- src/350_agent_cli.rs → `use crate :: agent_conscience :: AgentConscience ;`
- src/350_agent_cli.rs → `use crate :: invariant_types :: Invariant ;`
- src/040_refactor_constraints.rs → `use crate :: invariant_types :: * ;`
- src/410_dead_code_test_boundaries.rs → `use crate :: dead_code_call_graph :: { build_reverse_call_graph , CallGraph } ;`
- src/380_dead_code_doc_comment_parser.rs → `use crate :: dead_code_types :: { IntentMarker , IntentMap } ;`
- src/440_dead_code_confidence.rs → `use crate :: dead_code_types :: { ConfidenceLevel , DeadCodeCategory , DeadCodeItem } ;`
- src/020_gather_rust_files.rs → `use crate :: layer_utilities :: allow_analysis_dir ;`
- src/020_gather_rust_files.rs → `use crate :: layer_utilities :: resolve_source_root ;`
- src/admission_composition_artifact.rs → `use crate :: composition_rule :: { ComposedEffectState , CompositionResult , ConflictReason } ;`
- src/admission_composition_artifact.rs → `use crate :: effect_signature_schema :: EffectSignature ;`
- src/admission_composition_artifact.rs → `use crate :: composition_rule :: InvariantType ;`
- src/admission_composition_artifact.rs → `use crate :: composition_rule :: InvariantType ;`
- src/admission_composition_artifact.rs → `use crate :: composition_rule :: compose_batch ;`
- src/admission_composition_artifact.rs → `use crate :: effect_signature_schema :: * ;`
- src/610_action_impact_estimator.rs → `use crate :: quality_delta_calculator :: Metrics ;`
- src/610_action_impact_estimator.rs → `# [allow (unused_imports)] pub use crate :: quality_delta_calculator :: estimate_impact ;`
- src/610_action_impact_estimator.rs → `# [allow (unused_imports)] pub (crate) use crate :: correction_intelligence_report :: simulate_action ;`
- src/390_dead_code_call_graph.rs → `use crate :: dead_code_test_boundaries :: TestBoundaries ;`
- src/390_dead_code_call_graph.rs → `use crate :: types :: { CodeElement , ElementType , Language } ;`
- src/190_conscience_graph.rs → `use crate :: invariant_types :: * ;`
- src/160_invariant_integrator.rs → `use crate :: invariant_types :: * ;`
- src/160_invariant_integrator.rs → `use crate :: layer_inference :: { detect_layer_violations , infer_layers } ;`
- src/160_invariant_integrator.rs → `use crate :: path_detector :: PathDetector ;`
- src/160_invariant_integrator.rs → `use crate :: refactor_constraints :: generate_constraints ;`
- src/160_invariant_integrator.rs → `use crate :: semantic_detector :: SemanticDetector ;`
- src/160_invariant_integrator.rs → `use crate :: structural_detector :: StructuralDetector ;`
- src/160_invariant_integrator.rs → `use crate :: types :: { AnalysisResult , CallGraphNode } ;`
- src/160_invariant_integrator.rs → `use crate :: types :: { CodeElement , ElementType , Language , Visibility } ;`
- src/150_path_detector.rs → `use crate :: invariant_types :: * ;`
- src/150_path_detector.rs → `use crate :: scc_compressor :: SccCompression ;`
- src/180_invariant_reporter.rs → `use crate :: invariant_types :: * ;`
- src/180_invariant_reporter.rs → `use crate :: refactor_constraints :: RefactorConstraint ;`
- src/040_classify_symbol.rs → `use crate :: dead_code_call_graph :: { is_reachable , is_test_only , CallGraph } ;`
- src/040_classify_symbol.rs → `use crate :: dead_code_intent :: DeadCodePolicy ;`
- src/040_classify_symbol.rs → `use crate :: dead_code_test_boundaries :: TestBoundaries ;`
- src/040_classify_symbol.rs → `use crate :: dead_code_types :: { DeadCodeCategory , IntentMap } ;`
- src/510_quality_delta_types.rs → `use crate :: correction_plan_types :: ViolationType ;`
- src/070_layer_inference.rs → `use crate :: invariant_types :: LayerInfo ;`
- src/300_rust_parser.rs → `use crate :: types :: { AnalysisResult , CfgEdge , CfgNode , CodeElement , ElementType , FunctionCfg , Language , ModuleInfo , NodeType , Visibility , } ;`
- src/240_types.rs → `use crate :: invariant_types :: InvariantAnalysisResult ;`
- src/240_types.rs → `use crate :: refactor_constraints :: RefactorConstraint ;`
- src/590_rollback_criteria_builder.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: build_rollback_criteria ;`
- src/140_layer_core.rs → `# [allow (unused_imports)] pub use crate :: cluster_006 :: { layer_prefix_value , order_directories , collect_directory_moves , } ;`
- src/140_layer_core.rs → `# [allow (unused_imports)] pub use crate :: cluster_008 :: detect_layer_violations ;`
- src/140_layer_core.rs → `# [allow (unused_imports)] pub use crate :: cluster_001 :: { layer_constrained_sort , topo_sort_within } ;`
- src/140_layer_core.rs → `pub use crate :: cluster_008 :: sort_structural_items ;`
- src/110_structural_detector.rs → `use crate :: invariant_types :: * ;`
- src/110_structural_detector.rs → `use crate :: layer_inference :: infer_layers ;`
- src/110_structural_detector.rs → `use crate :: scc_compressor :: SccCompression ;`
- src/110_structural_detector.rs → `use crate :: types :: { AnalysisResult , ElementType } ;`
- src/170_layer_utilities.rs → `use crate :: cluster_001 :: run_analysis ;`
- src/170_layer_utilities.rs → `use crate :: cluster_010 :: gather_rust_files ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_actions :: recommend_action ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_call_graph :: { build_call_graph , classify_symbol , is_reachable } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_cli :: { DeadCodeRunConfig , is_test_path , merge_intent_map , reason_for_category } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_confidence :: { assign_confidence , Evidence } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_entrypoints :: { collect_entrypoints , collect_exports , is_public_api } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_attribute_parser :: { detect_test_modules , detect_test_symbols } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_intent :: detect_intent_signals ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_report :: { build_report , write_outputs , DeadCodeReportMetadata , DeadCodeReportWithMeta , } ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_test_boundaries :: TestBoundaries ;`
- src/170_layer_utilities.rs → `use crate :: dead_code_types :: { DeadCodeCategory , DeadCodeItem } ;`
- src/170_layer_utilities.rs → `use crate :: types :: { CodeElement , ElementType , Language , Visibility } ;`
- src/170_layer_utilities.rs → `# [allow (unused_imports)] pub use crate :: cluster_001 :: { build_file_layers , detect_layer , gather_julia_files , julia_entry_paths } ;`
- src/170_layer_utilities.rs → `# [allow (unused_imports)] pub use crate :: cluster_010 :: contains_tools ;`
- src/batch_admission.rs → `use crate :: admission_composition_artifact :: { generate_artifact , write_artifact } ;`
- src/batch_admission.rs → `use crate :: composition_rule :: compose_batch ;`
- src/batch_admission.rs → `use crate :: effect_signature_schema :: EffectSignature ;`
- src/batch_admission.rs → `use crate :: effect_signature_schema :: * ;`
- src/000_dependency_analysis.rs → `use crate :: cluster_010 :: { gather_rust_files , LayerResolver } ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: { LayerGraph , ReferenceDetail , UnresolvedDependency } ;`
- src/000_dependency_analysis.rs → `use crate :: file_ordering :: { build_dependency_map , build_entries , build_file_dag , detect_cycles , ordered_by_name , topological_sort , } ;`
- src/000_dependency_analysis.rs → `use crate :: layer_core :: layer_constrained_sort ;`
- src/000_dependency_analysis.rs → `use crate :: layer_utilities :: build_file_layers ;`
- src/000_dependency_analysis.rs → `use crate :: types :: FileOrderingResult ;`
- src/000_dependency_analysis.rs → `use crate :: utilities :: compress_path ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: naming_score_for_file ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: analyze_file_ordering ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: analyze_file_ordering ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: analyze_file_ordering ;`
- src/000_dependency_analysis.rs → `use crate :: cluster_006 :: layer_prefix_value ;`
- src/000_dependency_analysis.rs → `use crate :: cluster_001 :: { collect_julia_dependencies , JuliaTarget } ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: ReferenceDetail ;`
- src/000_dependency_analysis.rs → `use crate :: control_flow :: ControlFlowAnalyzer ;`
- src/000_dependency_analysis.rs → `use crate :: cohesion_analyzer :: FunctionCohesionAnalyzer ;`
- src/000_dependency_analysis.rs → `use crate :: dependency :: LayerGraph ;`
- src/000_dependency_analysis.rs → `use crate :: directory_analyzer :: DirectoryAnalyzer ;`
- src/000_dependency_analysis.rs → `use crate :: dot_exporter :: export_program_cfg_to_path ;`
- src/000_dependency_analysis.rs → `use crate :: julia_parser :: JuliaAnalyzer ;`
- src/000_dependency_analysis.rs → `use crate :: report :: ReportGenerator ;`
- src/000_dependency_analysis.rs → `use crate :: rust_parser :: RustAnalyzer ;`
- src/000_dependency_analysis.rs → `use crate :: types :: { AnalysisResult , FileOrderingResult } ;`
- src/000_dependency_analysis.rs → `use crate :: invariant_integrator :: InvariantDetector ;`
- src/000_dependency_analysis.rs → `use crate :: invariant_reporter ;`
- src/310_dot_exporter.rs → `# [allow (unused_imports)] pub use crate :: cluster_001 :: export_complete_program_dot ;`
- src/310_dot_exporter.rs → `pub use crate :: cluster_011 :: export_program_cfg_to_path ;`
- src/450_dead_code_actions.rs → `use crate :: dead_code_types :: { ConfidenceLevel , DeadCodeCategory , RecommendedAction } ;`
- src/570_correction_plan_generator.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: generate_correction_plan ;`
- src/400_dead_code_intent.rs → `pub use crate :: dead_code_attribute_parser :: detect_intent_signals ;`
- src/400_dead_code_intent.rs → `use crate :: dead_code_doc_comment_parser :: item_name ;`
- src/400_dead_code_intent.rs → `use crate :: dead_code_types :: { IntentMap , IntentMarker , IntentMetadata , IntentSource , } ;`
- src/280_file_ordering.rs → `use crate :: dependency :: build_module_map ;`
- src/280_file_ordering.rs → `use crate :: cluster_010 :: extract_dependencies ;`
- src/280_file_ordering.rs → `pub use crate :: cluster_001 :: { ordered_by_name , topological_sort } ;`
- src/280_file_ordering.rs → `pub use crate :: cluster_010 :: build_dependency_map ;`
- src/280_file_ordering.rs → `pub (crate) use crate :: cluster_001 :: build_entries ;`
- src/280_file_ordering.rs → `pub (crate) use crate :: cluster_011 :: build_file_dag ;`
- src/280_file_ordering.rs → `pub (crate) use crate :: cluster_001 :: detect_cycles ;`
- src/630_verification_policy_emitter.rs → `use crate :: verification_policy_types :: { QualityThresholds , VerificationCheck , VerificationPolicy , VerificationScope , } ;`
- src/500_dead_code_cli.rs → `use crate :: dead_code_intent :: DeadCodePolicy ;`
- src/500_dead_code_cli.rs → `use crate :: dead_code_types :: DeadCodeCategory ;`
- src/500_dead_code_cli.rs → `pub use crate :: layer_utilities :: run_dead_code_pipeline ;`
- src/580_verification_scope_planner.rs → `# [allow (unused_imports)] pub use crate :: correction_intelligence_report :: plan_verification_scope ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: collect_move_items ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: resolve_required_layer_path ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: write_cluster_batches ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: write_structural_batches ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: compute_move_metrics ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: path_common_prefix_len ;`
- src/220_utilities.rs → `# [allow (unused_imports)] pub use crate :: report :: collect_directory_files ;`
- src/220_utilities.rs → `# [doc = " Compress absolute paths to MMSB-relative format"] # [allow (unused_imports)] pub use crate :: report :: compress_path ;`
- src/130_semantic_detector.rs → `use crate :: invariant_types :: * ;`
- src/130_semantic_detector.rs → `use crate :: types :: { CodeElement , ElementType } ;`
- src/130_semantic_detector.rs → `use crate :: types :: { Language , Visibility } ;`
- src/230_dead_code_attribute_parser.rs → `use crate :: dead_code_doc_comment_parser :: { item_attrs , item_name , merge_doc_intent } ;`
- src/230_dead_code_attribute_parser.rs → `use crate :: dead_code_intent :: { check_planned_directory , collect_symbols , merge_intent_sources , planned_directory_intent , DeadCodePolicy , } ;`
- src/230_dead_code_attribute_parser.rs → `use crate :: dead_code_test_boundaries :: has_test_attr ;`
- src/230_dead_code_attribute_parser.rs → `use crate :: dead_code_types :: { IntentMap , IntentMarker , IntentMetadata , IntentSource , IntentTag } ;`

## Julia Layer Graph

No layers discovered.

