## Phase 4: Cohesion Improvements

Action: optional: improve cohesion by moving functions to better-fit modules.
Note: safe to defer unless you are actively refactoring.

- `parse_mmsb_latent_attr` from `src/230_dead_code_attribute_parser.rs` to `-`: cohesion 0.23 below threshold 0.60 (impact 0.37)
- `run_analysis` from `src/000_dependency_analysis.rs` to `-`: cohesion 0.20 below threshold 0.60 (impact 0.40)
- `scan_file_attributes` from `src/230_dead_code_attribute_parser.rs` to `-`: cohesion 0.23 below threshold 0.60 (impact 0.37)
- `write_intelligence_outputs_at` from `src/620_correction_plan_serializer.rs` to `-`: cohesion 0.35 below threshold 0.60 (impact 0.25)
- `build_directory_entry_map` from `src/000_dependency_analysis.rs` to `-`: cohesion 0.49 below threshold 0.60 (impact 0.11)
- `cluster_target_path` from `src/010_layer_utilities.rs` to `-`: cohesion 0.43 below threshold 0.60 (impact 0.17)
- `order_julia_files_by_dependency` from `src/000_dependency_analysis.rs` to `-`: cohesion 0.43 below threshold 0.60 (impact 0.17)
- `write_outputs` from `src/470_dead_code_report.rs` to `-`: cohesion 0.42 below threshold 0.60 (impact 0.18)
- `layer_constrained_sort` from `src/000_dependency_analysis.rs` to `-`: cohesion 0.57 below threshold 0.60 (impact 0.03)

