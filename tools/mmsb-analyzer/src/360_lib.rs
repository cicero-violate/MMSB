//! MMSB Analyzer Library
//!
//! Provides code analysis capabilities for Rust and Julia projects.

// PHASE 6.5: Admission Intelligence Formalization Foundation
pub mod effect_signature_schema;
pub mod composition_rule;
pub mod admission_composition_artifact;
pub mod batch_admission;

#[path = "030_is_cfg_test_item.rs"]
pub mod is_cfg_test_item;
#[path = "030_invariant_types.rs"]
pub mod invariant_types;
#[path = "040_refactor_constraints.rs"]
pub mod refactor_constraints;
#[path = "040_classify_symbol.rs"]
pub mod classify_symbol;
#[path = "050_scc_compressor.rs"]
pub mod scc_compressor;
#[path = "070_layer_inference.rs"]
pub mod layer_inference;
#[path = "090_fixpoint_solver.rs"]
pub mod fixpoint_solver;
#[path = "110_structural_detector.rs"]
pub mod structural_detector;
#[path = "130_semantic_detector.rs"]
pub mod semantic_detector;
#[path = "150_path_detector.rs"]
pub mod path_detector;
#[path = "160_invariant_integrator.rs"]
pub mod invariant_integrator;
#[path = "180_invariant_reporter.rs"]
pub mod invariant_reporter;
#[path = "190_conscience_graph.rs"]
pub mod conscience_graph;
#[path = "200_action_validator.rs"]
pub mod action_validator;
#[path = "210_agent_conscience.rs"]
pub mod agent_conscience;
#[path = "000_dependency_analysis.rs"]
pub mod cluster_001;
#[path = "010_layer_utilities.rs"]
pub mod cluster_008;
#[path = "060_module_resolution.rs"]
pub mod cluster_010;
#[path = "080_cluster_011.rs"]
pub mod cluster_011;
#[path = "120_cluster_006.rs"]
pub mod cluster_006;
#[path = "100_dependency.rs"]
pub mod dependency;
#[path = "140_layer_core.rs"]
pub mod layer_core;
#[path = "170_layer_utilities.rs"]
pub mod layer_utilities;
#[path = "220_utilities.rs"]
pub mod utilities;
#[path = "240_types.rs"]
pub mod types;
#[path = "250_cohesion_analyzer.rs"]
pub mod cohesion_analyzer;
#[path = "260_directory_analyzer.rs"]
pub mod directory_analyzer;
#[path = "270_control_flow.rs"]
pub mod control_flow;
#[path = "280_file_ordering.rs"]
pub mod file_ordering;
#[path = "290_julia_parser.rs"]
pub mod julia_parser;
#[path = "300_rust_parser.rs"]
pub mod rust_parser;
#[path = "310_dot_exporter.rs"]
pub mod dot_exporter;
#[path = "330_markdown_report.rs"]
pub mod report;
#[path = "370_dead_code_types.rs"]
pub mod dead_code_types;
#[path = "230_dead_code_attribute_parser.rs"]
pub mod dead_code_attribute_parser;
#[path = "380_dead_code_doc_comment_parser.rs"]
pub mod dead_code_doc_comment_parser;
#[path = "390_dead_code_call_graph.rs"]
pub mod dead_code_call_graph;
#[path = "400_dead_code_intent.rs"]
pub mod dead_code_intent;
#[path = "410_dead_code_test_boundaries.rs"]
pub mod dead_code_test_boundaries;
#[path = "420_dead_code_entrypoints.rs"]
pub mod dead_code_entrypoints;
#[path = "430_dead_code_classifier.rs"]
pub mod dead_code_classifier;
#[path = "440_dead_code_confidence.rs"]
pub mod dead_code_confidence;
#[path = "450_dead_code_actions.rs"]
pub mod dead_code_actions;
#[path = "470_dead_code_report.rs"]
pub mod dead_code_report;
#[path = "480_dead_code_filter.rs"]
pub mod dead_code_filter;
#[path = "500_dead_code_cli.rs"]
pub mod dead_code_cli;
#[path = "520_dead_code_policy.rs"]
pub mod dead_code_policy;
#[path = "540_dead_code_report_split.rs"]
pub mod dead_code_report_split;
#[path = "460_correction_plan_types.rs"]
pub mod correction_plan_types;
#[path = "490_verification_policy_types.rs"]
pub mod verification_policy_types;
#[path = "510_quality_delta_types.rs"]
pub mod quality_delta_types;
#[path = "530_violation_predictor.rs"]
pub mod violation_predictor;
#[path = "550_tier_classifier.rs"]
pub mod tier_classifier;
#[path = "560_confidence_scorer.rs"]
pub mod confidence_scorer;
#[path = "570_correction_plan_generator.rs"]
pub mod correction_plan_generator;
#[path = "580_verification_scope_planner.rs"]
pub mod verification_scope_planner;
#[path = "590_rollback_criteria_builder.rs"]
pub mod rollback_criteria_builder;
#[path = "600_quality_delta_calculator.rs"]
pub mod quality_delta_calculator;
#[path = "610_action_impact_estimator.rs"]
pub mod action_impact_estimator;
#[path = "620_correction_plan_serializer.rs"]
pub mod correction_plan_serializer;
#[path = "630_verification_policy_emitter.rs"]
pub mod verification_policy_emitter;
#[path = "640_correction_intelligence_report.rs"]
pub mod correction_intelligence_report;

pub use effect_signature_schema::{
    EffectSignature, ReadEffects, WriteEffects, StructuralTransitions,
    InvariantTouchpoints, ExecutorSurface, SCHEMA_VERSION,
};
pub use composition_rule::{
    compose_batch, CompositionResult, ComposedEffectState,
    ConflictReason, InvariantType,
};
pub use admission_composition_artifact::{
    generate_artifact, write_artifact, read_artifact,
    AdmissionCompositionArtifact, ARTIFACT_SCHEMA_VERSION,
};
pub use batch_admission::{admit_batch, AdmissionDecision};
pub use invariant_types::*;
pub use refactor_constraints::*;
pub use action_validator::{AgentAction, ConstraintViolation, ViolationSeverity};
pub use agent_conscience::{ActionPermission, AgentConscience};
pub use conscience_graph::{generate_conscience_map, generate_conscience_stats};
pub use dependency::{
    julia_entry_paths, order_julia_files_by_dependency, order_rust_files_by_dependency, LayerGraph,
    build_file_dependency_graph, analyze_file_ordering,
};
pub use types::AnalysisResult;
pub use cohesion_analyzer::FunctionCohesionAnalyzer;
pub use directory_analyzer::DirectoryAnalyzer;
pub use control_flow::ControlFlowAnalyzer;
pub use file_ordering::{DagCache, parallel_build_file_dag};
pub use julia_parser::JuliaAnalyzer;
pub use rust_parser::RustAnalyzer;
pub use dot_exporter::{export_complete_program_dot, export_program_cfg_to_path};
pub use report::ReportGenerator;
