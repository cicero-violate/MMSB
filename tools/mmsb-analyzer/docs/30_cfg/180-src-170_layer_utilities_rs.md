# CFG Group: src/170_layer_utilities.rs

## Function: `allow_analysis_dir`

- File: src/170_layer_utilities.rs
- Branches: 3
- Loops: 0
- Nodes: 18
- Edges: 20

```mermaid
flowchart TD
    allow_analysis_dir_0["ENTRY"]
    allow_analysis_dir_1["let name = dir . file_name () . and_then (| n | n . to_str ()) . unwrap_or ('')"]
    allow_analysis_dir_2["if name . starts_with ('.') || name == 'target' || name == 'node_modules'"]
    allow_analysis_dir_3["THEN BB"]
    allow_analysis_dir_4["return false"]
    allow_analysis_dir_5["EMPTY ELSE"]
    allow_analysis_dir_6["IF JOIN"]
    allow_analysis_dir_7["if let Ok (rel) = dir . strip_prefix (root)"]
    allow_analysis_dir_8["THEN BB"]
    allow_analysis_dir_9["if rel . components () . any (| c | { let s = c . as_os_str () . to_str () . ..."]
    allow_analysis_dir_10["THEN BB"]
    allow_analysis_dir_11["return false"]
    allow_analysis_dir_12["EMPTY ELSE"]
    allow_analysis_dir_13["IF JOIN"]
    allow_analysis_dir_14["EMPTY ELSE"]
    allow_analysis_dir_15["IF JOIN"]
    allow_analysis_dir_16["true"]
    allow_analysis_dir_17["EXIT"]
    allow_analysis_dir_0 --> allow_analysis_dir_1
    allow_analysis_dir_1 --> allow_analysis_dir_2
    allow_analysis_dir_2 --> allow_analysis_dir_3
    allow_analysis_dir_3 --> allow_analysis_dir_4
    allow_analysis_dir_2 --> allow_analysis_dir_5
    allow_analysis_dir_4 --> allow_analysis_dir_6
    allow_analysis_dir_5 --> allow_analysis_dir_6
    allow_analysis_dir_6 --> allow_analysis_dir_7
    allow_analysis_dir_7 --> allow_analysis_dir_8
    allow_analysis_dir_8 --> allow_analysis_dir_9
    allow_analysis_dir_9 --> allow_analysis_dir_10
    allow_analysis_dir_10 --> allow_analysis_dir_11
    allow_analysis_dir_9 --> allow_analysis_dir_12
    allow_analysis_dir_11 --> allow_analysis_dir_13
    allow_analysis_dir_12 --> allow_analysis_dir_13
    allow_analysis_dir_7 --> allow_analysis_dir_14
    allow_analysis_dir_13 --> allow_analysis_dir_15
    allow_analysis_dir_14 --> allow_analysis_dir_15
    allow_analysis_dir_15 --> allow_analysis_dir_16
    allow_analysis_dir_16 --> allow_analysis_dir_17
```

## Function: `main`

- File: src/170_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    main_0["ENTRY"]
    main_1["let args = Args :: parse ()"]
    main_2["let root_path = std :: env :: current_dir () ? . join (& args . root) . canonicalize () ?"]
    main_3["let output_path = std :: env :: current_dir () ? . join (& args . output) . canonicalize () . u..."]
    main_4["run_analysis (& root_path , & output_path , args . verbose , args . skip_juli..."]
    main_5["EXIT"]
    main_0 --> main_1
    main_1 --> main_2
    main_2 --> main_3
    main_3 --> main_4
    main_4 --> main_5
```

## Function: `resolve_source_root`

- File: src/170_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 9
- Edges: 9

```mermaid
flowchart TD
    resolve_source_root_0["ENTRY"]
    resolve_source_root_1["let src_candidate = root . join ('src')"]
    resolve_source_root_2["if src_candidate . exists () && src_candidate . is_dir ()"]
    resolve_source_root_3["THEN BB"]
    resolve_source_root_4["src_candidate"]
    resolve_source_root_5["ELSE BB"]
    resolve_source_root_6["{ root . to_path_buf () }"]
    resolve_source_root_7["IF JOIN"]
    resolve_source_root_8["EXIT"]
    resolve_source_root_0 --> resolve_source_root_1
    resolve_source_root_1 --> resolve_source_root_2
    resolve_source_root_2 --> resolve_source_root_3
    resolve_source_root_3 --> resolve_source_root_4
    resolve_source_root_2 --> resolve_source_root_5
    resolve_source_root_5 --> resolve_source_root_6
    resolve_source_root_4 --> resolve_source_root_7
    resolve_source_root_6 --> resolve_source_root_7
    resolve_source_root_7 --> resolve_source_root_8
```

## Function: `run_dead_code_pipeline`

- File: src/170_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 15
- Edges: 14

```mermaid
flowchart TD
    run_dead_code_pipeline_0["ENTRY"]
    run_dead_code_pipeline_1["let rust_files = gather_rust_files (& config . root)"]
    run_dead_code_pipeline_2["let mut intent_map = HashMap :: new ()"]
    run_dead_code_pipeline_3["let mut test_boundaries = TestBoundaries :: default ()"]
    run_dead_code_pipeline_4["for file in & rust_files { let intents = detect_intent_signals (file , config..."]
    run_dead_code_pipeline_5["let call_graph = build_call_graph (elements)"]
    run_dead_code_pipeline_6["let entrypoints = collect_entrypoints (elements , config . policy . as_ref ())"]
    run_dead_code_pipeline_7["let exports = collect_exports (& config . root)"]
    run_dead_code_pipeline_8["let mut items = Vec :: new ()"]
    run_dead_code_pipeline_9["for element in elements { if element . element_type != ElementType :: Functio..."]
    run_dead_code_pipeline_10["let metadata = DeadCodeReportMetadata { analyzer_version : env ! ('CARGO_PKG_VERSION') . to_..."]
    run_dead_code_pipeline_11["let report = build_report (chrono :: Local :: now () . to_rfc3339 () , items , metadata ,)"]
    run_dead_code_pipeline_12["write_outputs (& report , config) ?"]
    run_dead_code_pipeline_13["Ok (report)"]
    run_dead_code_pipeline_14["EXIT"]
    run_dead_code_pipeline_0 --> run_dead_code_pipeline_1
    run_dead_code_pipeline_1 --> run_dead_code_pipeline_2
    run_dead_code_pipeline_2 --> run_dead_code_pipeline_3
    run_dead_code_pipeline_3 --> run_dead_code_pipeline_4
    run_dead_code_pipeline_4 --> run_dead_code_pipeline_5
    run_dead_code_pipeline_5 --> run_dead_code_pipeline_6
    run_dead_code_pipeline_6 --> run_dead_code_pipeline_7
    run_dead_code_pipeline_7 --> run_dead_code_pipeline_8
    run_dead_code_pipeline_8 --> run_dead_code_pipeline_9
    run_dead_code_pipeline_9 --> run_dead_code_pipeline_10
    run_dead_code_pipeline_10 --> run_dead_code_pipeline_11
    run_dead_code_pipeline_11 --> run_dead_code_pipeline_12
    run_dead_code_pipeline_12 --> run_dead_code_pipeline_13
    run_dead_code_pipeline_13 --> run_dead_code_pipeline_14
```

