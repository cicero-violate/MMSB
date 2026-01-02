#[path = "030_invariant_types.rs"]
mod invariant_types;
#[path = "040_refactor_constraints.rs"]
mod refactor_constraints;
#[path = "050_scc_compressor.rs"]
mod scc_compressor;
#[path = "070_layer_inference.rs"]
mod layer_inference;
#[path = "090_fixpoint_solver.rs"]
mod fixpoint_solver;
#[path = "110_structural_detector.rs"]
mod structural_detector;
#[path = "130_semantic_detector.rs"]
mod semantic_detector;
#[path = "150_path_detector.rs"]
mod path_detector;
#[path = "160_invariant_integrator.rs"]
mod invariant_integrator;
#[path = "180_invariant_reporter.rs"]
mod invariant_reporter;
#[path = "190_conscience_graph.rs"]
mod conscience_graph;
#[path = "200_action_validator.rs"]
mod action_validator;
#[path = "210_agent_conscience.rs"]
mod agent_conscience;
#[path = "350_agent_cli.rs"]
mod agent_cli;
#[path = "000_dependency_analysis.rs"]
mod cluster_001;
#[path = "010_layer_utilities.rs"]
mod cluster_008;
#[path = "060_module_resolution.rs"]
mod cluster_010;
#[path = "080_cluster_011.rs"]
mod cluster_011;
#[path = "120_cluster_006.rs"]
mod cluster_006;
#[path = "270_control_flow.rs"]
mod control_flow;
#[path = "260_directory_analyzer.rs"]
mod directory_analyzer;
#[path = "140_layer_core.rs"]
mod layer_core;
#[path = "170_layer_utilities.rs"]
mod layer_utilities;
#[path = "100_dependency.rs"]
mod dependency;
#[path = "220_utilities.rs"]
mod utilities;
#[path = "310_dot_exporter.rs"]
mod dot_exporter;
#[path = "250_cohesion_analyzer.rs"]
mod cohesion_analyzer;
#[path = "280_file_ordering.rs"]
mod file_ordering;
#[path = "290_julia_parser.rs"]
mod julia_parser;
#[path = "330_markdown_report.rs"]
mod report;
#[path = "370_dead_code_types.rs"]
mod dead_code_types;
#[path = "230_dead_code_attribute_parser.rs"]
mod dead_code_attribute_parser;
#[path = "380_dead_code_doc_comment_parser.rs"]
mod dead_code_doc_comment_parser;
#[path = "390_dead_code_call_graph.rs"]
mod dead_code_call_graph;
#[path = "400_dead_code_intent.rs"]
mod dead_code_intent;
#[path = "410_dead_code_test_boundaries.rs"]
mod dead_code_test_boundaries;
#[path = "420_dead_code_entrypoints.rs"]
mod dead_code_entrypoints;
#[path = "430_dead_code_classifier.rs"]
mod dead_code_classifier;
#[path = "440_dead_code_confidence.rs"]
mod dead_code_confidence;
#[path = "450_dead_code_actions.rs"]
mod dead_code_actions;
#[path = "470_dead_code_report.rs"]
mod dead_code_report;
#[path = "480_dead_code_filter.rs"]
mod dead_code_filter;
#[path = "500_dead_code_cli.rs"]
mod dead_code_cli;
#[path = "520_dead_code_policy.rs"]
mod dead_code_policy;
#[path = "540_dead_code_report_split.rs"]
mod dead_code_report_split;
#[path = "460_correction_plan_types.rs"]
mod correction_plan_types;
#[path = "490_verification_policy_types.rs"]
mod verification_policy_types;
#[path = "510_quality_delta_types.rs"]
mod quality_delta_types;
#[path = "530_violation_predictor.rs"]
mod violation_predictor;
#[path = "550_tier_classifier.rs"]
mod tier_classifier;
#[path = "560_confidence_scorer.rs"]
mod confidence_scorer;
#[path = "570_correction_plan_generator.rs"]
mod correction_plan_generator;
#[path = "580_verification_scope_planner.rs"]
mod verification_scope_planner;
#[path = "590_rollback_criteria_builder.rs"]
mod rollback_criteria_builder;
#[path = "600_quality_delta_calculator.rs"]
mod quality_delta_calculator;
#[path = "610_action_impact_estimator.rs"]
mod action_impact_estimator;
#[path = "620_correction_plan_serializer.rs"]
mod correction_plan_serializer;
#[path = "630_verification_policy_emitter.rs"]
mod verification_policy_emitter;
#[path = "640_correction_intelligence_report.rs"]
mod correction_intelligence_report;
#[path = "300_rust_parser.rs"]
mod rust_parser;
#[path = "240_types.rs"]
mod types;

use anyhow::Result;

fn main() -> Result<()> {
    // Check if running in agent mode
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "agent" {
        // Remove "agent" from args and run agent CLI
        return agent_cli::run_agent_cli();
    }

    // Normal analyzer mode
    crate::layer_utilities::main()
}
