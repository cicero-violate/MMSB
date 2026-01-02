# Conscience Map

## Overview

This map shows which functions are protected by mechanical constraints.
Functions with blocking invariants cannot be refactored without violating proven properties.

**Total Functions**: 209

**Protected Functions**: 209 (100.0%)

---

## Functions by Protection Level

### `item_attrs` (80% protected)

**File**: `src/410_dead_code_test_boundaries.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **DegreeStable(in=4, out=0)**: Degree: in=4, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(68, size=1)**: Member of SCC 68 (size: 1)
- 🔒 ✓ **SCC(148, size=1)**: Member of SCC 148 (size: 1)

### `main` (80% protected)

**File**: `src/340_main.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(86, size=1)**: Member of SCC 86 (size: 1)
- 🔒 ✓ **SCC(141, size=1)**: Member of SCC 141 (size: 1)

### `layer_prefix_value` (80% protected)

**File**: `src/120_cluster_006.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **DegreeStable(in=13, out=0)**: Degree: in=13, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(14, size=1)**: Member of SCC 14 (size: 1)
- 🔒 ✓ **SCC(54, size=1)**: Member of SCC 54 (size: 1)

### `is_reachable` (63% protected)

**File**: `src/430_dead_code_classifier.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(71, size=1)**: Member of SCC 71 (size: 1)
- 🔒 ✓ **SCC(149, size=1)**: Member of SCC 149 (size: 1)

### `extract_identifiers` (77% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(129, size=1)**: Member of SCC 129 (size: 1)
- 🔒 ✓ **SCC(170, size=1)**: Member of SCC 170 (size: 1)

### `export_json` (83% protected)

**File**: `src/180_invariant_reporter.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(121, size=1)**: Member of SCC 121 (size: 1)

### `cyclomatic_complexity` (83% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(65, size=1)**: Member of SCC 65 (size: 1)

### `julia_entry_paths` (83% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(34, size=1)**: Member of SCC 34 (size: 1)

### `export_complete_program_dot` (83% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(38, size=1)**: Member of SCC 38 (size: 1)

### `node_style` (83% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(64, size=1)**: Member of SCC 64 (size: 1)

### `write_artifact` (83% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(4, size=1)**: Member of SCC 4 (size: 1)

### `generate_conscience_map` (83% protected)

**File**: `src/190_conscience_graph.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(123, size=1)**: Member of SCC 123 (size: 1)

### `classify_tier` (83% protected)

**File**: `src/550_tier_classifier.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(159, size=1)**: Member of SCC 159 (size: 1)

### `build_basic_report` (83% protected)

**File**: `src/470_dead_code_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(151, size=1)**: Member of SCC 151 (size: 1)

### `compute_cohesion_score` (71% protected)

**File**: `src/120_cluster_006.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(85, size=1)**: Member of SCC 85 (size: 1)

### `collect_roots` (100% protected)

**File**: `src/100_dependency.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(81, size=1)**: Member of SCC 81 (size: 1)

### `resolve_path` (83% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(78, size=1)**: Member of SCC 78 (size: 1)

### `build_state` (83% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(166, size=1)**: Member of SCC 166 (size: 1)

### `export_constraints_json` (83% protected)

**File**: `src/180_invariant_reporter.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(122, size=1)**: Member of SCC 122 (size: 1)

### `compute_confidence` (71% protected)

**File**: `src/560_confidence_scorer.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(160, size=1)**: Member of SCC 160 (size: 1)

### `parallel_build_file_dag` (83% protected)

**File**: `src/280_file_ordering.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(130, size=1)**: Member of SCC 130 (size: 1)

### `read_artifact` (83% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(5, size=1)**: Member of SCC 5 (size: 1)

### `extract_attribute_value` (83% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=0)**: Degree: in=0, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(127, size=1)**: Member of SCC 127 (size: 1)

### `insert_sorted` (100% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(50, size=1)**: Member of SCC 50 (size: 1)

### `write_cluster_batches` (100% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(138, size=1)**: Member of SCC 138 (size: 1)

### `is_test_attribute_line` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(174, size=1)**: Member of SCC 174 (size: 1)

### `generate_artifact` (80% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=3)**: Degree: in=0, out=3 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(3, size=1)**: Member of SCC 3 (size: 1)

### `module_name_from_path` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(200, size=1)**: Member of SCC 200 (size: 1)

### `topo_sort_within` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(13, size=1)**: Member of SCC 13 (size: 1)

### `average_confidence` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(191, size=1)**: Member of SCC 191 (size: 1)

### `has_test_attr` (66% protected)

**File**: `src/410_dead_code_test_boundaries.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(103, size=1)**: Member of SCC 103 (size: 1)

### `build_module_root_map` (80% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(76, size=1)**: Member of SCC 76 (size: 1)

### `compute_summary` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=4, out=0)**: Degree: in=4, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(167, size=1)**: Member of SCC 167 (size: 1)

### `validate_action` (80% protected)

**File**: `src/200_action_validator.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(125, size=1)**: Member of SCC 125 (size: 1)

### `detect_layer` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(18, size=1)**: Member of SCC 18 (size: 1)

### `assign_confidence` (80% protected)

**File**: `src/440_dead_code_confidence.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(96, size=1)**: Member of SCC 96 (size: 1)

### `write_intelligence_outputs` (80% protected)

**File**: `src/620_correction_plan_serializer.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(165, size=1)**: Member of SCC 165 (size: 1)

### `parse_list` (66% protected)

**File**: `src/520_dead_code_policy.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=3, out=0)**: Degree: in=3, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(156, size=1)**: Member of SCC 156 (size: 1)

### `collect_rust_dependencies` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(31, size=1)**: Member of SCC 31 (size: 1)

### `generate_intelligence_report` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=8)**: Degree: in=0, out=8 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(213, size=1)**: Member of SCC 213 (size: 1)

### `collect_julia_dependencies` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(33, size=1)**: Member of SCC 33 (size: 1)

### `find_element_file` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(205, size=1)**: Member of SCC 205 (size: 1)

### `sort_structural_items` (100% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=3)**: Degree: in=0, out=3 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(67, size=1)**: Member of SCC 67 (size: 1)

### `layer_adheres` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(57, size=1)**: Member of SCC 57 (size: 1)

### `collect_invariants_touched` (80% protected)

**File**: `src/composition_rule.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(6, size=1)**: Member of SCC 6 (size: 1)

### `compress_path` (80% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=5, out=0)**: Degree: in=5, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(131, size=1)**: Member of SCC 131 (size: 1)

### `resolve_source_root` (80% protected)

**File**: `src/170_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(44, size=1)**: Member of SCC 44 (size: 1)

### `write_structural_batches` (100% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(137, size=1)**: Member of SCC 137 (size: 1)

### `generate_phase2_cluster_slice` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=5)**: Degree: in=0, out=5 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(199, size=1)**: Member of SCC 199 (size: 1)

### `write_report` (80% protected)

**File**: `src/470_dead_code_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(91, size=1)**: Member of SCC 91 (size: 1)

### `filter_visibility_report` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(182, size=1)**: Member of SCC 182 (size: 1)

### `build_result` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=6)**: Degree: in=0, out=6 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(53, size=1)**: Member of SCC 53 (size: 1)

### `estimate_verification_time` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(186, size=1)**: Member of SCC 186 (size: 1)

### `scan_intent_tags` (80% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=4)**: Degree: in=0, out=4 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(128, size=1)**: Member of SCC 128 (size: 1)

### `symbol_exists` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(206, size=1)**: Member of SCC 206 (size: 1)

### `detect_violations` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(36, size=1)**: Member of SCC 36 (size: 1)

### `strip_numeric_prefix` (80% protected)

**File**: `src/120_cluster_006.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(84, size=1)**: Member of SCC 84 (size: 1)

### `compare_dir_layers` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(55, size=1)**: Member of SCC 55 (size: 1)

### `scan_file_attributes` (80% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=3)**: Degree: in=0, out=3 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(126, size=1)**: Member of SCC 126 (size: 1)

### `collect_use_tree_idents` (100% protected)

**File**: `src/420_dead_code_entrypoints.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(98, size=1)**: Member of SCC 98 (size: 1)

### `detect_layer_violation` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(59, size=1)**: Member of SCC 59 (size: 1)

### `collect_symbols` (80% protected)

**File**: `src/400_dead_code_intent.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(108, size=1)**: Member of SCC 108 (size: 1)

### `order_rust_files_by_dependency` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=3)**: Degree: in=0, out=3 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(32, size=1)**: Member of SCC 32 (size: 1)

### `affected_files` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(188, size=1)**: Member of SCC 188 (size: 1)

### `collect_move_items` (80% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=5)**: Degree: in=0, out=5 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(140, size=1)**: Member of SCC 140 (size: 1)

### `serialize_correction_plan` (80% protected)

**File**: `src/620_correction_plan_serializer.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(161, size=1)**: Member of SCC 161 (size: 1)

### `check_planned_directory` (80% protected)

**File**: `src/400_dead_code_intent.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(109, size=1)**: Member of SCC 109 (size: 1)

### `merge_doc_intent` (80% protected)

**File**: `src/380_dead_code_doc_comment_parser.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(111, size=1)**: Member of SCC 111 (size: 1)

### `move_violates_invariant` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(207, size=1)**: Member of SCC 207 (size: 1)

### `load_invariants` (80% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=4, out=0)**: Degree: in=4, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(142, size=1)**: Member of SCC 142 (size: 1)

### `recommend_action` (80% protected)

**File**: `src/450_dead_code_actions.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(94, size=1)**: Member of SCC 94 (size: 1)

### `merge_intent_map` (100% protected)

**File**: `src/500_dead_code_cli.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(106, size=1)**: Member of SCC 106 (size: 1)

### `is_function_signature_line` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=4, out=0)**: Degree: in=4, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(171, size=1)**: Member of SCC 171 (size: 1)

### `compare_path_components` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(56, size=1)**: Member of SCC 56 (size: 1)

### `gather_julia_files` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(35, size=1)**: Member of SCC 35 (size: 1)

### `is_identifier_candidate` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(169, size=1)**: Member of SCC 169 (size: 1)

### `should_exclude_from_analysis` (80% protected)

**File**: `src/480_dead_code_filter.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(152, size=1)**: Member of SCC 152 (size: 1)

### `extract_layer` (80% protected)

**File**: `src/200_action_validator.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(124, size=1)**: Member of SCC 124 (size: 1)

### `extract_critical_tests` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(184, size=1)**: Member of SCC 184 (size: 1)

### `parse_cluster_members` (66% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(60, size=1)**: Member of SCC 60 (size: 1)

### `estimate_fix_time` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(190, size=1)**: Member of SCC 190 (size: 1)

### `action_module_path` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(197, size=1)**: Member of SCC 197 (size: 1)

### `collect_cluster_plans` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(63, size=1)**: Member of SCC 63 (size: 1)

### `item_name` (80% protected)

**File**: `src/380_dead_code_doc_comment_parser.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=3, out=0)**: Degree: in=3, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(114, size=1)**: Member of SCC 114 (size: 1)

### `analyze_file_ordering` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=7)**: Degree: in=0, out=7 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(37, size=1)**: Member of SCC 37 (size: 1)

### `generate_canonical_name` (80% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(135, size=1)**: Member of SCC 135 (size: 1)

### `export_program_cfg_to_path` (80% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(40, size=1)**: Member of SCC 40 (size: 1)

### `is_mmsb_main` (66% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(48, size=1)**: Member of SCC 48 (size: 1)

### `topological_sort` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(12, size=1)**: Member of SCC 12 (size: 1)

### `collect_naming_warnings` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(7)**: Layer 7 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(29, size=1)**: Member of SCC 29 (size: 1)

### `compose_batch` (80% protected)

**File**: `src/composition_rule.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(9, size=1)**: Member of SCC 9 (size: 1)

### `is_test_path` (66% protected)

**File**: `src/500_dead_code_cli.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(102, size=1)**: Member of SCC 102 (size: 1)

### `build_directory_dag` (80% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(79, size=1)**: Member of SCC 79 (size: 1)

### `common_root` (80% protected)

**File**: `src/120_cluster_006.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(82, size=1)**: Member of SCC 82 (size: 1)

### `action_visibility` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(192, size=1)**: Member of SCC 192 (size: 1)

### `generate_constraints` (80% protected)

**File**: `src/040_refactor_constraints.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(41, size=1)**: Member of SCC 41 (size: 1)

### `action_module` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(187, size=1)**: Member of SCC 187 (size: 1)

### `filter_path_coherence_report` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(168, size=1)**: Member of SCC 168 (size: 1)

### `build_module_map` (80% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(77, size=1)**: Member of SCC 77 (size: 1)

### `compute_reachability` (66% protected)

**File**: `src/390_dead_code_call_graph.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(70, size=1)**: Member of SCC 70 (size: 1)

### `order_julia_files_by_dependency` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(39, size=1)**: Member of SCC 39 (size: 1)

### `parse_phase2_cluster_plan` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(183, size=1)**: Member of SCC 183 (size: 1)

### `parse_bool` (66% protected)

**File**: `src/520_dead_code_policy.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(155, size=1)**: Member of SCC 155 (size: 1)

### `build_entries` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(10, size=1)**: Member of SCC 10 (size: 1)

### `emit_verification_policy` (80% protected)

**File**: `src/630_verification_policy_emitter.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(163, size=1)**: Member of SCC 163 (size: 1)

### `ordered_by_name` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=4, out=0)**: Degree: in=4, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(11, size=1)**: Member of SCC 11 (size: 1)

### `collect_directory_files` (100% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(132, size=1)**: Member of SCC 132 (size: 1)

### `path_common_prefix_len` (80% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(133, size=1)**: Member of SCC 133 (size: 1)

### `top_items` (80% protected)

**File**: `src/540_dead_code_report_split.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(88, size=1)**: Member of SCC 88 (size: 1)

### `default_confidence` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(202, size=1)**: Member of SCC 202 (size: 1)

### `load_policy` (80% protected)

**File**: `src/520_dead_code_policy.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(158, size=1)**: Member of SCC 158 (size: 1)

### `marker_from_str` (80% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(116, size=1)**: Member of SCC 116 (size: 1)

### `project_invariants_touched` (80% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(0, size=1)**: Member of SCC 0 (size: 1)

### `build_file_dependency_graph` (80% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=2)**: Degree: in=0, out=2 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(80, size=1)**: Member of SCC 80 (size: 1)

### `layer_rank_map` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(49, size=1)**: Member of SCC 49 (size: 1)

### `plan_options` (80% protected)

**File**: `src/540_dead_code_report_split.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(87, size=1)**: Member of SCC 87 (size: 1)

### `write_admission_preflight_report` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(5)**: Layer 5 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(181, size=1)**: Member of SCC 181 (size: 1)

### `run_dead_code_pipeline` (80% protected)

**File**: `src/170_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(4)**: Layer 4 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=17)**: Degree: in=0, out=17 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(120, size=1)**: Member of SCC 120 (size: 1)

### `action_function` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(194, size=1)**: Member of SCC 194 (size: 1)

### `is_core_module_path` (66% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(61, size=1)**: Member of SCC 61 (size: 1)

### `treat_public_as_entrypoint` (80% protected)

**File**: `src/420_dead_code_entrypoints.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(100, size=1)**: Member of SCC 100 (size: 1)

### `action_refs` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(196, size=1)**: Member of SCC 196 (size: 1)

### `contains_tools` (80% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(75, size=1)**: Member of SCC 75 (size: 1)

### `detect_cycles` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(16, size=1)**: Member of SCC 16 (size: 1)

### `naming_score_for_file` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(28, size=1)**: Member of SCC 28 (size: 1)

### `compute_move_metrics` (66% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(134, size=1)**: Member of SCC 134 (size: 1)

### `run_agent_cli` (80% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=4)**: Degree: in=0, out=4 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(147, size=1)**: Member of SCC 147 (size: 1)

### `build_reverse_call_graph` (80% protected)

**File**: `src/390_dead_code_call_graph.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(72, size=1)**: Member of SCC 72 (size: 1)

### `build_file_dag` (80% protected)

**File**: `src/080_cluster_011.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=3, out=0)**: Degree: in=3, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(17, size=1)**: Member of SCC 17 (size: 1)

### `build_report` (80% protected)

**File**: `src/470_dead_code_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(93, size=1)**: Member of SCC 93 (size: 1)

### `adjacency_from_edges` (80% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(52, size=1)**: Member of SCC 52 (size: 1)

### `filter_dead_code_elements` (80% protected)

**File**: `src/480_dead_code_filter.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(154, size=1)**: Member of SCC 154 (size: 1)

### `allow_analysis_dir` (80% protected)

**File**: `src/170_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(43, size=1)**: Member of SCC 43 (size: 1)

### `calculate_quality_delta` (66% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(204, size=1)**: Member of SCC 204 (size: 1)

### `is_public_api` (66% protected)

**File**: `src/420_dead_code_entrypoints.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(95, size=1)**: Member of SCC 95 (size: 1)

### `find_test_callers` (80% protected)

**File**: `src/410_dead_code_test_boundaries.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(150, size=1)**: Member of SCC 150 (size: 1)

### `action_target_layer` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(193, size=1)**: Member of SCC 193 (size: 1)

### `order_directories` (80% protected)

**File**: `src/120_cluster_006.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(83, size=1)**: Member of SCC 83 (size: 1)

### `merge_intent_sources` (80% protected)

**File**: `src/400_dead_code_intent.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(107, size=1)**: Member of SCC 107 (size: 1)

### `simulate_action` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(210, size=1)**: Member of SCC 210 (size: 1)

### `detect_latent_markers` (80% protected)

**File**: `src/380_dead_code_doc_comment_parser.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(112, size=1)**: Member of SCC 112 (size: 1)

### `normalize_module_name` (66% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(20, size=1)**: Member of SCC 20 (size: 1)

### `action_symbol` (80% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(195, size=1)**: Member of SCC 195 (size: 1)

### `reason_for_category` (80% protected)

**File**: `src/500_dead_code_cli.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(97, size=1)**: Member of SCC 97 (size: 1)

### `build_call_graph` (80% protected)

**File**: `src/390_dead_code_call_graph.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=0)**: Degree: in=2, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(42, size=1)**: Member of SCC 42 (size: 1)

### `collect_directory_moves` (80% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=0, out=1)**: Degree: in=0, out=1 (proven from graph)
- 🔒 ✓ **Root**: Root node (called by no other functions)
- 🔒 ✓ **SCC(136, size=1)**: Member of SCC 136 (size: 1)

### `rust_entry_paths` (80% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(30, size=1)**: Member of SCC 30 (size: 1)

### `project_conflict_reason` (80% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(0)**: Layer 0 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=0)**: Degree: in=1, out=0 (proven from graph)
- 🔒 ✓ **Leaf**: Leaf node (calls no other functions)
- 🔒 ✓ **SCC(2, size=1)**: Member of SCC 2 (size: 1)

### `function_in_impl_block` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **SCC(176, size=1)**: Member of SCC 176 (size: 1)

### `scan_doc_comments` (75% protected)

**File**: `src/211_dead_code_doc_comment_scanner.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=3)**: Degree: in=2, out=3 (proven from graph)
- 🔒 ✓ **SCC(115, size=1)**: Member of SCC 115 (size: 1)

### `is_cfg_test_item` (60% protected)

**File**: `src/030_is_cfg_test_item.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **SCC(69, size=1)**: Member of SCC 69 (size: 1)

### `write_summary_markdown` (60% protected)

**File**: `src/540_dead_code_report_split.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(90, size=1)**: Member of SCC 90 (size: 1)

### `build_dependency_map` (60% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(5)**: Layer 5 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(26, size=1)**: Member of SCC 26 (size: 1)

### `collect_exports` (60% protected)

**File**: `src/420_dead_code_entrypoints.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(99, size=1)**: Member of SCC 99 (size: 1)

### `detect_test_symbols` (75% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(104, size=1)**: Member of SCC 104 (size: 1)

### `function_signature_found` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(177, size=1)**: Member of SCC 177 (size: 1)

### `serialize_correction_plans` (60% protected)

**File**: `src/620_correction_plan_serializer.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(162, size=1)**: Member of SCC 162 (size: 1)

### `structural_layer_value` (75% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=6, out=1)**: Degree: in=6, out=1 (proven from graph)
- 🔒 ✓ **SCC(58, size=1)**: Member of SCC 58 (size: 1)

### `query_function` (60% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(145, size=1)**: Member of SCC 145 (size: 1)

### `extract_function_block_from_contents` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(172, size=1)**: Member of SCC 172 (size: 1)

### `cluster_target_path` (75% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(62, size=1)**: Member of SCC 62 (size: 1)

### `find_private_dependencies` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(173, size=1)**: Member of SCC 173 (size: 1)

### `build_file_layers` (75% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **SCC(19, size=1)**: Member of SCC 19 (size: 1)

### `layer_constrained_sort` (75% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(15, size=1)**: Member of SCC 15 (size: 1)

### `is_test_scoped_function` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(175, size=1)**: Member of SCC 175 (size: 1)

### `project_state` (60% protected)

**File**: `src/admission_composition_artifact.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(1, size=1)**: Member of SCC 1 (size: 1)

### `collect_latent_attrs` (75% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **SCC(117, size=1)**: Member of SCC 117 (size: 1)

### `generate_correction_plan` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=12)**: Degree: in=2, out=12 (proven from graph)
- 🔒 ✓ **SCC(198, size=1)**: Member of SCC 198 (size: 1)

### `gather_rust_files` (75% protected)

**File**: `src/020_gather_rust_files.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=2)**: Degree: in=2, out=2 (proven from graph)
- 🔒 ✓ **SCC(45, size=1)**: Member of SCC 45 (size: 1)

### `find_callers` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(208, size=1)**: Member of SCC 208 (size: 1)

### `is_layer_violation` (60% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(47, size=1)**: Member of SCC 47 (size: 1)

### `generate_admission_preflight` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(4)**: Layer 4 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(180, size=1)**: Member of SCC 180 (size: 1)

### `show_stats` (60% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(143, size=1)**: Member of SCC 143 (size: 1)

### `extract_rust_dependencies` (75% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(24, size=1)**: Member of SCC 24 (size: 1)

### `detect_test_modules` (60% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(105, size=1)**: Member of SCC 105 (size: 1)

### `resolve_module_name` (75% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=4, out=1)**: Degree: in=4, out=1 (proven from graph)
- 🔒 ✓ **SCC(22, size=1)**: Member of SCC 22 (size: 1)

### `list_invariants` (60% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(144, size=1)**: Member of SCC 144 (size: 1)

### `plan_verification_scope` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=3)**: Degree: in=2, out=3 (proven from graph)
- 🔒 ✓ **SCC(189, size=1)**: Member of SCC 189 (size: 1)

### `extract_dependencies` (75% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(4)**: Layer 4 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(25, size=1)**: Member of SCC 25 (size: 1)

### `collect_excluded_symbols` (60% protected)

**File**: `src/480_dead_code_filter.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(153, size=1)**: Member of SCC 153 (size: 1)

### `check_action` (60% protected)

**File**: `src/350_agent_cli.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(146, size=1)**: Member of SCC 146 (size: 1)

### `classify_symbol` (75% protected)

**File**: `src/040_classify_symbol.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(74, size=1)**: Member of SCC 74 (size: 1)

### `write_plan_markdown` (75% protected)

**File**: `src/540_dead_code_report_split.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(89, size=1)**: Member of SCC 89 (size: 1)

### `resolve_module` (75% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=5, out=1)**: Degree: in=5, out=1 (proven from graph)
- 🔒 ✓ **SCC(21, size=1)**: Member of SCC 21 (size: 1)

### `topo_sort` (60% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(51, size=1)**: Member of SCC 51 (size: 1)

### `structural_cmp` (75% protected)

**File**: `src/010_layer_utilities.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=4)**: Degree: in=1, out=4 (proven from graph)
- 🔒 ✓ **SCC(66, size=1)**: Member of SCC 66 (size: 1)

### `fill_prediction_confidence` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(203, size=1)**: Member of SCC 203 (size: 1)

### `build_directory_entry_map` (75% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(6)**: Layer 6 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=9)**: Degree: in=1, out=9 (proven from graph)
- 🔒 ✓ **SCC(27, size=1)**: Member of SCC 27 (size: 1)

### `is_test_only` (50% protected)

**File**: `src/390_dead_code_call_graph.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(73, size=1)**: Member of SCC 73 (size: 1)

### `planned_directory_intent` (75% protected)

**File**: `src/400_dead_code_intent.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(110, size=1)**: Member of SCC 110 (size: 1)

### `collect_entrypoints` (60% protected)

**File**: `src/420_dead_code_entrypoints.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(101, size=1)**: Member of SCC 101 (size: 1)

### `find_reference_files` (60% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(209, size=1)**: Member of SCC 209 (size: 1)

### `evaluate_move_admission` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=4)**: Degree: in=1, out=4 (proven from graph)
- 🔒 ✓ **SCC(179, size=1)**: Member of SCC 179 (size: 1)

### `compose_into_state` (75% protected)

**File**: `src/composition_rule.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(7, size=1)**: Member of SCC 7 (size: 1)

### `augment_path_coherence_strategies` (100% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(201, size=1)**: Member of SCC 201 (size: 1)

### `find_function_definition_candidates` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(178, size=1)**: Member of SCC 178 (size: 1)

### `predict_violations` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=4)**: Degree: in=1, out=4 (proven from graph)
- 🔒 ✓ **SCC(211, size=1)**: Member of SCC 211 (size: 1)

### `run_analysis` (75% protected)

**File**: `src/000_dependency_analysis.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=5)**: Degree: in=1, out=5 (proven from graph)
- 🔒 ✓ **SCC(46, size=1)**: Member of SCC 46 (size: 1)

### `detect_intent_signals` (75% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=5)**: Degree: in=1, out=5 (proven from graph)
- 🔒 ✓ **SCC(119, size=1)**: Member of SCC 119 (size: 1)

### `build_rollback_criteria` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=1)**: Degree: in=2, out=1 (proven from graph)
- 🔒 ✓ **SCC(185, size=1)**: Member of SCC 185 (size: 1)

### `estimate_impact` (75% protected)

**File**: `src/640_correction_intelligence_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(212, size=1)**: Member of SCC 212 (size: 1)

### `check_conflicts` (60% protected)

**File**: `src/composition_rule.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(8, size=1)**: Member of SCC 8 (size: 1)

### `extract_julia_dependencies` (75% protected)

**File**: `src/060_module_resolution.rs`

- 🔒 ✓ **LayerFixed(3)**: Layer 3 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=5)**: Degree: in=1, out=5 (proven from graph)
- 🔒 ✓ **SCC(23, size=1)**: Member of SCC 23 (size: 1)

### `write_outputs` (75% protected)

**File**: `src/470_dead_code_report.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=3)**: Degree: in=1, out=3 (proven from graph)
- 🔒 ✓ **SCC(92, size=1)**: Member of SCC 92 (size: 1)

### `write_intelligence_outputs_at` (75% protected)

**File**: `src/620_correction_plan_serializer.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(164, size=1)**: Member of SCC 164 (size: 1)

### `extract_doc_markers` (60% protected)

**File**: `src/380_dead_code_doc_comment_parser.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=1)**: Degree: in=1, out=1 (proven from graph)
- 🔒 ✓ **SCC(113, size=1)**: Member of SCC 113 (size: 1)

### `resolve_required_layer_path` (75% protected)

**File**: `src/330_markdown_report.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=2)**: Degree: in=1, out=2 (proven from graph)
- 🔒 ✓ **SCC(139, size=1)**: Member of SCC 139 (size: 1)

### `parse_mmsb_latent_attr` (60% protected)

**File**: `src/230_dead_code_attribute_parser.rs`

- 🔒 ✓ **LayerFixed(2)**: Layer 2 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=2, out=3)**: Degree: in=2, out=3 (proven from graph)
- 🔒 ✓ **SCC(118, size=1)**: Member of SCC 118 (size: 1)

### `parse_policy` (60% protected)

**File**: `src/520_dead_code_policy.rs`

- 🔒 ✓ **LayerFixed(1)**: Layer 1 assignment is proven from call graph
- 🔒 ✓ **DegreeStable(in=1, out=4)**: Degree: in=1, out=4 (proven from graph)
- 🔒 ✓ **SCC(157, size=1)**: Member of SCC 157 (size: 1)

---

## Legend

- ✓ **PROVEN**: Mathematical certainty from graph topology
- ◆ **EMPIRICAL**: Observed across multiple paths (high confidence)
- ? **HEURISTIC**: Name-based guess (LOW CONFIDENCE - verify manually)

- 🔒 **Blocking**: Constraint mechanically enforced

