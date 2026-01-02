# Module Imports

## src/000_dependency_analysis.rs (000_dependency_analysis.rs)

Module `000_dependency_analysis`

- `anyhow :: { Context , Result }`
- `crate :: cluster_001 :: { collect_julia_dependencies , JuliaTarget }`
- `crate :: cluster_006 :: layer_prefix_value`
- `crate :: cluster_010 :: { gather_rust_files , LayerResolver }`
- `crate :: cohesion_analyzer :: FunctionCohesionAnalyzer`
- `crate :: control_flow :: ControlFlowAnalyzer`
- `crate :: dependency :: LayerGraph`
- `crate :: dependency :: ReferenceDetail`
- `crate :: dependency :: analyze_file_ordering`
- `crate :: dependency :: naming_score_for_file`
- `crate :: dependency :: { LayerGraph , ReferenceDetail , UnresolvedDependency }`
- `crate :: directory_analyzer :: DirectoryAnalyzer`
- `crate :: dot_exporter :: export_program_cfg_to_path`
- `crate :: file_ordering :: { build_dependency_map , build_entries , build_file_dag , detect_cycles , ordered_by_name , topological_sort , }`
- `crate :: invariant_integrator :: InvariantDetector`
- `crate :: invariant_reporter`
- `crate :: julia_parser :: JuliaAnalyzer`
- `crate :: layer_core :: layer_constrained_sort`
- `crate :: layer_utilities :: build_file_layers`
- `crate :: report :: ReportGenerator`
- `crate :: rust_parser :: RustAnalyzer`
- `crate :: types :: FileOrderingResult`
- `crate :: types :: { AnalysisResult , FileOrderingResult }`
- `crate :: utilities :: compress_path`
- `once_cell :: sync :: Lazy`
- `petgraph :: Direction`
- `petgraph :: algo :: tarjan_scc`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `regex :: Regex`
- `std :: collections :: HashMap`
- `std :: collections :: HashSet`
- `std :: collections :: VecDeque`
- `std :: collections :: { BTreeMap , BTreeSet , HashMap , HashSet }`
- `std :: fmt :: Write`
- `std :: fs`
- `std :: fs :: { create_dir_all , write }`
- `std :: path :: { Path , PathBuf }`
- `super :: *`
- `syn :: visit :: Visit`
- `syn :: { ItemUse , UseTree }`
- `walkdir :: WalkDir`

## src/010_layer_utilities.rs (010_layer_utilities.rs)

Module `010_layer_utilities`

- `anyhow :: Result`
- `crate :: dependency :: { LayerEdge , LayerGraph , ReferenceDetail , UnresolvedDependency , }`
- `crate :: types :: { FileLayerViolation , NodeType }`
- `petgraph :: graph :: DiGraph`
- `petgraph :: visit :: EdgeRef`
- `std :: cmp :: Ordering`
- `std :: collections :: HashMap`
- `std :: collections :: { BTreeMap , BTreeSet , HashMap , VecDeque }`
- `std :: path :: PathBuf`
- `std :: path :: { Path , PathBuf }`

## src/020_gather_rust_files.rs (020_gather_rust_files.rs)

Module `020_gather_rust_files`

- `crate :: layer_utilities :: allow_analysis_dir`
- `crate :: layer_utilities :: resolve_source_root`
- `std :: path :: Path`
- `std :: path :: PathBuf`
- `walkdir :: WalkDir`

## src/030_invariant_types.rs (030_invariant_types.rs)

Module `030_invariant_types`

- `serde :: { Deserialize , Serialize }`
- `std :: collections :: { HashMap , HashSet }`
- `std :: fmt`
- `super :: *`

## src/030_is_cfg_test_item.rs (030_is_cfg_test_item.rs)

Module `030_is_cfg_test_item`

- `crate :: dead_code_doc_comment_parser :: item_attrs`
- `syn :: Item`

## src/040_classify_symbol.rs (040_classify_symbol.rs)

Module `040_classify_symbol`

- `crate :: dead_code_call_graph :: { is_reachable , is_test_only , CallGraph }`
- `crate :: dead_code_intent :: DeadCodePolicy`
- `crate :: dead_code_test_boundaries :: TestBoundaries`
- `crate :: dead_code_types :: { DeadCodeCategory , IntentMap }`
- `std :: collections :: HashSet`

## src/040_refactor_constraints.rs (040_refactor_constraints.rs)

Module `040_refactor_constraints`

- `crate :: invariant_types :: *`
- `serde :: { Deserialize , Serialize }`
- `std :: fmt`
- `super :: *`

## src/050_scc_compressor.rs (050_scc_compressor.rs)

Module `050_scc_compressor`

- `petgraph :: algo :: tarjan_scc`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `petgraph :: visit :: EdgeRef`
- `std :: collections :: HashMap`
- `super :: *`

## src/060_module_resolution.rs (060_module_resolution.rs)

Module `060_module_resolution`

- `anyhow :: { Context , Result }`
- `crate :: dependency :: RootState`
- `crate :: layer_utilities :: resolve_source_root`
- `once_cell :: sync :: Lazy`
- `regex :: Regex`
- `std :: collections :: { BTreeSet , HashMap , HashSet }`
- `std :: fs`
- `std :: path :: { Path , PathBuf }`
- `syn :: ItemUse`
- `syn :: visit :: Visit`
- `walkdir :: WalkDir`

## src/070_layer_inference.rs (070_layer_inference.rs)

Module `070_layer_inference`

- `crate :: invariant_types :: LayerInfo`
- `petgraph :: Direction`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `petgraph :: visit :: EdgeRef`
- `std :: collections :: HashMap`
- `super :: *`

## src/080_cluster_011.rs (080_cluster_011.rs)

Module `080_cluster_011`

- `anyhow :: Result`
- `crate :: types :: ProgramCFG`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `std :: collections :: { HashMap , HashSet }`
- `std :: path :: { Path , PathBuf }`

## src/090_fixpoint_solver.rs (090_fixpoint_solver.rs)

Module `090_fixpoint_solver`

- `petgraph :: Direction`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `std :: collections :: { HashMap , HashSet }`
- `super :: *`

## src/100_dependency.rs (100_dependency.rs)

Module `100_dependency`

- `std :: collections :: BTreeSet`
- `std :: path :: PathBuf`
- `syn :: UseTree`

## src/110_structural_detector.rs (110_structural_detector.rs)

Module `110_structural_detector`

- `crate :: invariant_types :: *`
- `crate :: layer_inference :: infer_layers`
- `crate :: scc_compressor :: SccCompression`
- `crate :: types :: { AnalysisResult , ElementType }`
- `petgraph :: Direction`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `std :: collections :: HashMap`
- `super :: *`

## src/120_cluster_006.rs (120_cluster_006.rs)

Module `120_cluster_006`

- `crate :: cluster_008 :: FunctionInfo`
- `once_cell :: sync :: Lazy`
- `regex :: Regex`
- `std :: collections :: { BTreeSet , HashMap , HashSet }`
- `std :: path :: { Path , PathBuf }`

## src/130_semantic_detector.rs (130_semantic_detector.rs)

Module `130_semantic_detector`

- `crate :: invariant_types :: *`
- `crate :: types :: { CodeElement , ElementType }`
- `crate :: types :: { Language , Visibility }`
- `regex :: Regex`
- `super :: *`

## src/150_path_detector.rs (150_path_detector.rs)

Module `150_path_detector`

- `crate :: invariant_types :: *`
- `crate :: scc_compressor :: SccCompression`
- `petgraph :: Direction`
- `petgraph :: algo :: all_simple_paths`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `std :: collections :: HashSet`
- `super :: *`

## src/160_invariant_integrator.rs (160_invariant_integrator.rs)

Module `160_invariant_integrator`

- `crate :: invariant_types :: *`
- `crate :: layer_inference :: { detect_layer_violations , infer_layers }`
- `crate :: path_detector :: PathDetector`
- `crate :: refactor_constraints :: generate_constraints`
- `crate :: semantic_detector :: SemanticDetector`
- `crate :: structural_detector :: StructuralDetector`
- `crate :: types :: { AnalysisResult , CallGraphNode }`
- `crate :: types :: { CodeElement , ElementType , Language , Visibility }`
- `petgraph :: graph :: DiGraph`
- `std :: collections :: HashMap`
- `super :: *`

## src/170_layer_utilities.rs (170_layer_utilities.rs)

Module `170_layer_utilities`

- `anyhow :: Result`
- `clap :: Parser`
- `crate :: cluster_001 :: run_analysis`
- `crate :: cluster_010 :: gather_rust_files`
- `crate :: dead_code_actions :: recommend_action`
- `crate :: dead_code_attribute_parser :: { detect_test_modules , detect_test_symbols }`
- `crate :: dead_code_call_graph :: { build_call_graph , classify_symbol , is_reachable }`
- `crate :: dead_code_cli :: { DeadCodeRunConfig , is_test_path , merge_intent_map , reason_for_category }`
- `crate :: dead_code_confidence :: { assign_confidence , Evidence }`
- `crate :: dead_code_entrypoints :: { collect_entrypoints , collect_exports , is_public_api }`
- `crate :: dead_code_intent :: detect_intent_signals`
- `crate :: dead_code_report :: { build_report , write_outputs , DeadCodeReportMetadata , DeadCodeReportWithMeta , }`
- `crate :: dead_code_test_boundaries :: TestBoundaries`
- `crate :: dead_code_types :: { DeadCodeCategory , DeadCodeItem }`
- `crate :: types :: { CodeElement , ElementType , Language , Visibility }`
- `std :: collections :: HashMap`
- `std :: path :: { Path , PathBuf }`

## src/180_invariant_reporter.rs (180_invariant_reporter.rs)

Module `180_invariant_reporter`

- `crate :: invariant_types :: *`
- `crate :: refactor_constraints :: RefactorConstraint`
- `serde_json`
- `std :: collections :: HashMap`
- `std :: fs`
- `std :: path :: Path`
- `super :: *`

## src/190_conscience_graph.rs (190_conscience_graph.rs)

Module `190_conscience_graph`

- `crate :: invariant_types :: *`
- `std :: collections :: HashMap`
- `std :: path :: Path`
- `super :: *`

## src/200_action_validator.rs (200_action_validator.rs)

Module `200_action_validator`

- `crate :: invariant_types :: InvariantStrength`
- `crate :: refactor_constraints :: RefactorConstraint`
- `serde :: { Deserialize , Serialize }`
- `std :: path :: PathBuf`
- `super :: *`

## src/210_agent_conscience.rs (210_agent_conscience.rs)

Module `210_agent_conscience`

- `crate :: action_validator :: { validate_action , AgentAction , ConstraintViolation }`
- `crate :: invariant_types :: { Invariant , InvariantStrength }`
- `crate :: invariant_types :: { InvariantKind , SemanticInvariant }`
- `crate :: invariant_types :: { InvariantKind , StructuralInvariant }`
- `crate :: refactor_constraints :: { from_invariant , RefactorConstraint }`
- `serde :: { Deserialize , Serialize }`
- `std :: path :: { Path , PathBuf }`
- `super :: *`

## src/211_dead_code_doc_comment_scanner.rs (211_dead_code_doc_comment_scanner.rs)

Module `211_dead_code_doc_comment_scanner`

- `crate :: dead_code_doc_comment_parser :: extract_doc_markers`
- `crate :: dead_code_doc_comment_parser :: item_attrs`
- `crate :: dead_code_doc_comment_parser :: item_name`
- `crate :: dead_code_types :: IntentMarker`
- `std :: collections :: HashMap`
- `std :: path :: Path`

## src/230_dead_code_attribute_parser.rs (230_dead_code_attribute_parser.rs)

Module `230_dead_code_attribute_parser`

- `# [doc = " Attribute parsing for dead code intent markers."] use std :: collections :: { HashMap , HashSet }`
- `crate :: dead_code_doc_comment_parser :: { item_attrs , item_name , merge_doc_intent }`
- `crate :: dead_code_intent :: { check_planned_directory , collect_symbols , merge_intent_sources , planned_directory_intent , DeadCodePolicy , }`
- `crate :: dead_code_test_boundaries :: has_test_attr`
- `crate :: dead_code_types :: { IntentMap , IntentMarker , IntentMetadata , IntentSource , IntentTag }`
- `std :: path :: Path`
- `syn :: { Attribute , Item }`

## src/240_types.rs (240_types.rs)

Module `240_types`

- `crate :: invariant_types :: InvariantAnalysisResult`
- `crate :: refactor_constraints :: RefactorConstraint`
- `serde :: { Deserialize , Serialize }`
- `std :: collections :: HashMap`
- `std :: path :: PathBuf`

## src/250_cohesion_analyzer.rs (250_cohesion_analyzer.rs)

Module `250_cohesion_analyzer`

- `anyhow :: Result`
- `crate :: cluster_006 :: compute_cohesion_score`
- `crate :: cluster_008 :: { detect_layer_violation , FunctionInfo }`
- `crate :: types :: { AnalysisResult , ElementType }`
- `crate :: types :: { CallAnalysis , FunctionCluster , FunctionPlacement , PlacementStatus }`
- `std :: collections :: { BTreeMap , HashMap , HashSet }`
- `std :: path :: PathBuf`

## src/260_directory_analyzer.rs (260_directory_analyzer.rs)

Module `260_directory_analyzer`

- `anyhow :: Result`
- `crate :: dependency :: detect_layer`
- `crate :: layer_utilities :: allow_analysis_dir`
- `crate :: types :: DirectoryAnalysis`
- `std :: fs`
- `std :: path :: { Path , PathBuf }`

## src/270_control_flow.rs (270_control_flow.rs)

Module `270_control_flow`

- `crate :: types :: *`
- `crate :: utilities :: compress_path`
- `petgraph :: dot :: Dot`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `std :: collections :: HashMap`

## src/280_file_ordering.rs (280_file_ordering.rs)

Module `280_file_ordering`

- `(crate) use crate :: cluster_001 :: build_entries`
- `(crate) use crate :: cluster_001 :: detect_cycles`
- `(crate) use crate :: cluster_011 :: build_file_dag`
- `anyhow :: Result`
- `crate :: cluster_010 :: extract_dependencies`
- `crate :: dependency :: build_module_map`
- `petgraph :: graph :: { DiGraph , NodeIndex }`
- `petgraph :: visit :: EdgeRef`
- `rayon :: prelude :: *`
- `std :: collections :: { HashMap , HashSet }`
- `std :: fs`
- `std :: path :: PathBuf`
- `std :: time :: SystemTime`

## src/290_julia_parser.rs (290_julia_parser.rs)

Module `290_julia_parser`

- `anyhow :: { anyhow , Context , Result }`
- `crate :: types :: *`
- `once_cell :: sync :: Lazy`
- `regex :: Regex`
- `std :: env`
- `std :: fs`
- `std :: path :: { Path , PathBuf }`
- `std :: process :: { Command , Stdio }`
- `std :: sync :: atomic :: { AtomicBool , Ordering }`

## src/300_rust_parser.rs (300_rust_parser.rs)

Module `300_rust_parser`

- `anyhow :: { Context , Result }`
- `crate :: types :: { AnalysisResult , CfgEdge , CfgNode , CodeElement , ElementType , FunctionCfg , Language , ModuleInfo , NodeType , Visibility , }`
- `std :: fs`
- `std :: path :: { Path , PathBuf }`
- `syn :: visit :: Visit`
- `syn :: { ItemEnum , ItemFn , ItemImpl , ItemMod , ItemStruct , ItemTrait , ItemUse }`

## src/330_markdown_report.rs (330_markdown_report.rs)

Module `330_markdown_report`

- `crate :: action_validator :: check_move_allowed`
- `crate :: cluster_006 :: strip_numeric_prefix`
- `crate :: cluster_008 :: collect_cluster_plans`
- `crate :: control_flow :: ControlFlowAnalyzer`
- `crate :: dependency :: { LayerGraph , build_directory_entry_map , build_file_dependency_graph , collect_naming_warnings }`
- `crate :: file_ordering :: DirectoryMove`
- `crate :: layer_core :: { sort_structural_items }`
- `crate :: refactor_constraints :: RefactorConstraint`
- `crate :: types :: { AnalysisResult , CallGraphNode , CodeElement , DirectoryAnalysis , ElementType , FileOrderingResult , FunctionCfg , FunctionCluster , FunctionPlacement , Language , PlacementStatus , Visibility , }`
- `crate :: types :: { ElementType , Language , Visibility }`
- `std :: cmp :: Ordering`
- `std :: collections :: HashMap`
- `std :: collections :: { BTreeMap , BTreeSet , HashMap , HashSet }`
- `std :: fs`
- `std :: path :: PathBuf`
- `std :: path :: { Path , PathBuf }`
- `walkdir :: WalkDir`

## src/340_main.rs (340_main.rs)

Module `340_main`

- `anyhow :: Result`

## src/350_agent_cli.rs (350_agent_cli.rs)

Module `350_agent_cli`

- `anyhow :: Result`
- `clap :: Parser`
- `crate :: action_validator :: AgentAction`
- `crate :: agent_conscience :: AgentConscience`
- `crate :: invariant_types :: Invariant`
- `std :: path :: PathBuf`
- `super :: *`

## src/370_dead_code_types.rs (370_dead_code_types.rs)

Module `370_dead_code_types`

- `serde :: { Deserialize , Serialize }`
- `std :: collections :: HashMap`
- `std :: path :: PathBuf`

## src/380_dead_code_doc_comment_parser.rs (380_dead_code_doc_comment_parser.rs)

Module `380_dead_code_doc_comment_parser`

- `crate :: dead_code_types :: { IntentMarker , IntentMap }`
- `std :: collections :: { HashMap , HashSet }`
- `syn :: { Attribute , Item , Meta , MetaNameValue }`

## src/390_dead_code_call_graph.rs (390_dead_code_call_graph.rs)

Module `390_dead_code_call_graph`

- `crate :: dead_code_test_boundaries :: TestBoundaries`
- `crate :: types :: { CodeElement , ElementType , Language }`
- `std :: collections :: { HashMap , HashSet , VecDeque }`

## src/400_dead_code_intent.rs (400_dead_code_intent.rs)

Module `400_dead_code_intent`

- `crate :: dead_code_doc_comment_parser :: item_name`
- `crate :: dead_code_types :: { IntentMap , IntentMarker , IntentMetadata , IntentSource , }`
- `std :: collections :: HashMap`
- `std :: path :: { Path , PathBuf }`

## src/410_dead_code_test_boundaries.rs (410_dead_code_test_boundaries.rs)

Module `410_dead_code_test_boundaries`

- `crate :: dead_code_call_graph :: { build_reverse_call_graph , CallGraph }`
- `std :: collections :: { HashSet , VecDeque }`
- `std :: path :: PathBuf`
- `syn :: { Attribute , Item }`

## src/420_dead_code_entrypoints.rs (420_dead_code_entrypoints.rs)

Module `420_dead_code_entrypoints`

- `crate :: dead_code_intent :: DeadCodePolicy`
- `crate :: types :: { CodeElement , ElementType , Visibility }`
- `std :: collections :: HashSet`
- `std :: path :: Path`
- `walkdir :: WalkDir`

## src/430_dead_code_classifier.rs (430_dead_code_classifier.rs)

Module `430_dead_code_classifier`

- `crate :: dead_code_call_graph :: CallGraph`
- `std :: collections :: HashSet`

## src/440_dead_code_confidence.rs (440_dead_code_confidence.rs)

Module `440_dead_code_confidence`

- `crate :: dead_code_types :: { ConfidenceLevel , DeadCodeCategory , DeadCodeItem }`

## src/450_dead_code_actions.rs (450_dead_code_actions.rs)

Module `450_dead_code_actions`

- `crate :: dead_code_types :: { ConfidenceLevel , DeadCodeCategory , RecommendedAction }`

## src/460_correction_plan_types.rs (460_correction_plan_types.rs)

Module `460_correction_plan_types`

- `crate :: types :: Visibility`
- `serde :: { Deserialize , Serialize }`
- `std :: path :: PathBuf`

## src/470_dead_code_report.rs (470_dead_code_report.rs)

Module `470_dead_code_report`

- `anyhow :: Result`
- `crate :: dead_code_cli :: DeadCodeRunConfig`
- `crate :: dead_code_report_split :: { write_plan_markdown , write_summary_markdown }`
- `crate :: dead_code_types :: { DeadCodeItem , DeadCodeReport , DeadCodeSummary }`
- `serde :: { Deserialize , Serialize }`
- `std :: path :: Path`

## src/480_dead_code_filter.rs (480_dead_code_filter.rs)

Module `480_dead_code_filter`

- `crate :: dead_code_report :: DeadCodeReportWithMeta`
- `crate :: dead_code_types :: { DeadCodeCategory }`
- `crate :: types :: CodeElement`
- `std :: collections :: HashSet`

## src/490_verification_policy_types.rs (490_verification_policy_types.rs)

Module `490_verification_policy_types`

- `serde :: { Deserialize , Serialize }`
- `std :: path :: PathBuf`

## src/500_dead_code_cli.rs (500_dead_code_cli.rs)

Module `500_dead_code_cli`

- `crate :: dead_code_intent :: DeadCodePolicy`
- `crate :: dead_code_types :: DeadCodeCategory`
- `std :: collections :: HashMap`
- `std :: path :: { Path , PathBuf }`

## src/510_quality_delta_types.rs (510_quality_delta_types.rs)

Module `510_quality_delta_types`

- `crate :: correction_plan_types :: ViolationType`
- `serde :: { Deserialize , Serialize }`

## src/520_dead_code_policy.rs (520_dead_code_policy.rs)

Module `520_dead_code_policy`

- `crate :: dead_code_intent :: DeadCodePolicy`
- `std :: path :: Path`

## src/540_dead_code_report_split.rs (540_dead_code_report_split.rs)

Module `540_dead_code_report_split`

- `crate :: dead_code_report :: DeadCodeReportWithMeta`
- `crate :: dead_code_types :: { DeadCodeCategory , DeadCodeItem , RecommendedAction }`
- `std :: path :: Path`

## src/550_tier_classifier.rs (550_tier_classifier.rs)

Module `550_tier_classifier`

- `crate :: correction_plan_types :: { ErrorTier , Severity , ViolationPrediction , ViolationType }`

## src/560_confidence_scorer.rs (560_confidence_scorer.rs)

Module `560_confidence_scorer`

- `crate :: correction_plan_types :: { ViolationPrediction , ViolationType }`

## src/610_action_impact_estimator.rs (610_action_impact_estimator.rs)

Module `610_action_impact_estimator`

- `# [allow (unused_imports)] pub (crate) use crate :: correction_intelligence_report :: simulate_action`
- `crate :: quality_delta_calculator :: Metrics`

## src/620_correction_plan_serializer.rs (620_correction_plan_serializer.rs)

Module `620_correction_plan_serializer`

- `# [doc = " Serialize correction plans to JSON values."] use std :: path :: Path`
- `crate :: correction_intelligence_report :: CorrectionIntelligenceReport`
- `crate :: correction_plan_types :: { CorrectionPlan , CorrectionStrategy }`
- `crate :: quality_delta_types :: RollbackCriteria`
- `crate :: verification_policy_emitter :: emit_verification_policy`
- `crate :: verification_policy_types :: { QualityThresholds , VerificationCheck , VerificationPolicy , VerificationScope }`
- `serde_json :: { json , Value }`

## src/630_verification_policy_emitter.rs (630_verification_policy_emitter.rs)

Module `630_verification_policy_emitter`

- `crate :: verification_policy_types :: { QualityThresholds , VerificationCheck , VerificationPolicy , VerificationScope , }`
- `serde_json :: json`
- `std :: path :: Path`

## src/640_correction_intelligence_report.rs (640_correction_intelligence_report.rs)

Module `640_correction_intelligence_report`

- `crate :: action_impact_estimator :: AnalysisState`
- `crate :: action_impact_estimator :: AnalysisState as ImpactState`
- `crate :: correction_plan_types :: VisibilityPlanOption`
- `crate :: correction_plan_types :: { CorrectionPlan , CorrectionStrategy , ErrorTier , RefactorAction , Severity , ViolationPrediction , ViolationType , }`
- `crate :: invariant_types :: InvariantAnalysisResult`
- `crate :: quality_delta_calculator :: Metrics`
- `crate :: quality_delta_types :: RollbackCondition`
- `crate :: quality_delta_types :: { QualityDelta , RollbackCriteria }`
- `crate :: tier_classifier :: classify_tier`
- `crate :: types :: { AnalysisResult , CallGraphNode , CodeElement }`
- `crate :: verification_policy_types :: VerificationCheck`
- `crate :: verification_policy_types :: VerificationPolicy`
- `crate :: verification_policy_types :: VerificationScope`
- `regex :: Regex`
- `serde :: { Deserialize , Serialize }`
- `std :: collections :: HashMap`
- `std :: path :: { Path , PathBuf }`
- `std :: { collections :: HashSet , fs }`

## src/admission_composition_artifact.rs (root)

Module `admission_composition_artifact`

- `crate :: composition_rule :: InvariantType`
- `crate :: composition_rule :: compose_batch`
- `crate :: composition_rule :: { ComposedEffectState , CompositionResult , ConflictReason }`
- `crate :: effect_signature_schema :: *`
- `crate :: effect_signature_schema :: EffectSignature`
- `serde :: { Deserialize , Serialize }`
- `std :: collections :: BTreeSet`
- `std :: path :: Path`
- `std :: path :: PathBuf`
- `super :: *`

## src/batch_admission.rs (root)

Module `batch_admission`

- `crate :: admission_composition_artifact :: { generate_artifact , write_artifact }`
- `crate :: composition_rule :: compose_batch`
- `crate :: effect_signature_schema :: *`
- `crate :: effect_signature_schema :: EffectSignature`
- `std :: collections :: BTreeSet`
- `std :: path :: PathBuf`
- `std :: path :: { Path , PathBuf }`
- `super :: *`
- `tempfile :: TempDir`

## src/composition_rule.rs (root)

Module `composition_rule`

- `crate :: effect_signature_schema :: *`
- `serde :: { Deserialize , Serialize }`
- `std :: collections :: { BTreeMap , BTreeSet }`
- `std :: path :: PathBuf`
- `super :: *`

## src/effect_signature_schema.rs (root)

Module `effect_signature_schema`

- `serde :: { Deserialize , Serialize }`
- `std :: collections :: BTreeSet`
- `std :: path :: PathBuf`
- `super :: *`

