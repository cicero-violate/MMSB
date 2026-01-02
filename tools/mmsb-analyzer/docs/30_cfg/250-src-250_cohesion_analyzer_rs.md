# CFG Group: src/250_cohesion_analyzer.rs

## Function: `build_call_analysis`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    build_call_analysis_0["ENTRY"]
    build_call_analysis_1["let mut intra_file_calls = 0usize"]
    build_call_analysis_2["let mut inter_file_calls : BTreeMap < PathBuf , usize > = BTreeMap :: new ()"]
    build_call_analysis_3["for (callee_idx , count) in outgoing { let callee = & functions [* callee_idx..."]
    build_call_analysis_4["let mut calls_from_same_file = 0usize"]
    build_call_analysis_5["let mut calls_from_other_files : BTreeMap < PathBuf , usize > = BTreeMap :: new ()"]
    build_call_analysis_6["for (caller_idx , count) in incoming { let caller = & functions [* caller_idx..."]
    build_call_analysis_7["let (same_file_type_refs , other_file_type_refs) = compute_type_coupling (func , file_types , all_types)"]
    build_call_analysis_8["CallAnalysis { intra_file_calls , inter_file_calls : inter_file_calls . into_..."]
    build_call_analysis_9["EXIT"]
    build_call_analysis_0 --> build_call_analysis_1
    build_call_analysis_1 --> build_call_analysis_2
    build_call_analysis_2 --> build_call_analysis_3
    build_call_analysis_3 --> build_call_analysis_4
    build_call_analysis_4 --> build_call_analysis_5
    build_call_analysis_5 --> build_call_analysis_6
    build_call_analysis_6 --> build_call_analysis_7
    build_call_analysis_7 --> build_call_analysis_8
    build_call_analysis_8 --> build_call_analysis_9
```

## Function: `build_call_edges`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    build_call_edges_0["ENTRY"]
    build_call_edges_1["let functions = collect_functions (result)"]
    build_call_edges_2["let name_map = build_name_map (& functions)"]
    build_call_edges_3["let mut outgoing_counts : Vec < HashMap < usize , usize > > = vec ! [HashMap :: new () ; functions . len ()]"]
    build_call_edges_4["let mut incoming_counts : Vec < HashMap < usize , usize > > = vec ! [HashMap :: new () ; functions . len ()]"]
    build_call_edges_5["for (idx , func) in functions . iter () . enumerate () { for call in & func ...."]
    build_call_edges_6["(functions , outgoing_counts , incoming_counts)"]
    build_call_edges_7["EXIT"]
    build_call_edges_0 --> build_call_edges_1
    build_call_edges_1 --> build_call_edges_2
    build_call_edges_2 --> build_call_edges_3
    build_call_edges_3 --> build_call_edges_4
    build_call_edges_4 --> build_call_edges_5
    build_call_edges_5 --> build_call_edges_6
    build_call_edges_6 --> build_call_edges_7
```

## Function: `build_function_layers`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    build_function_layers_0["ENTRY"]
    build_function_layers_1["let mut map = HashMap :: new ()"]
    build_function_layers_2["for func in functions { map . entry (func . file_path . clone ()) . or_insert..."]
    build_function_layers_3["map"]
    build_function_layers_4["EXIT"]
    build_function_layers_0 --> build_function_layers_1
    build_function_layers_1 --> build_function_layers_2
    build_function_layers_2 --> build_function_layers_3
    build_function_layers_3 --> build_function_layers_4
```

## Function: `build_name_map`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    build_name_map_0["ENTRY"]
    build_name_map_1["let mut map : HashMap < String , Vec < usize > > = HashMap :: new ()"]
    build_name_map_2["for (idx , func) in functions . iter () . enumerate () { map . entry (func . ..."]
    build_name_map_3["map"]
    build_name_map_4["EXIT"]
    build_name_map_0 --> build_name_map_1
    build_name_map_1 --> build_name_map_2
    build_name_map_2 --> build_name_map_3
    build_name_map_3 --> build_name_map_4
```

## Function: `build_type_maps`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    build_type_maps_0["ENTRY"]
    build_type_maps_1["let mut file_types : HashMap < String , HashSet < String > > = HashMap :: new ()"]
    build_type_maps_2["let mut all_types : HashSet < String > = HashSet :: new ()"]
    build_type_maps_3["for elem in & result . elements { if matches ! (elem . element_type , Element..."]
    build_type_maps_4["(file_types , all_types)"]
    build_type_maps_5["EXIT"]
    build_type_maps_0 --> build_type_maps_1
    build_type_maps_1 --> build_type_maps_2
    build_type_maps_2 --> build_type_maps_3
    build_type_maps_3 --> build_type_maps_4
    build_type_maps_4 --> build_type_maps_5
```

## Function: `build_undirected_graph`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    build_undirected_graph_0["ENTRY"]
    build_undirected_graph_1["let n = outgoing_counts . len ()"]
    build_undirected_graph_2["let mut edge_weights : HashMap < (usize , usize) , usize > = HashMap :: new ()"]
    build_undirected_graph_3["for (src , outgoing) in outgoing_counts . iter () . enumerate () { for (dst ,..."]
    build_undirected_graph_4["let mut neighbors = vec ! [Vec :: new () ; n]"]
    build_undirected_graph_5["let mut degrees = vec ! [0usize ; n]"]
    build_undirected_graph_6["let mut total_weight = 0usize"]
    build_undirected_graph_7["for ((a , b) , weight) in edge_weights { if a == b { continue ; } neighbors [..."]
    build_undirected_graph_8["(neighbors , degrees , total_weight)"]
    build_undirected_graph_9["EXIT"]
    build_undirected_graph_0 --> build_undirected_graph_1
    build_undirected_graph_1 --> build_undirected_graph_2
    build_undirected_graph_2 --> build_undirected_graph_3
    build_undirected_graph_3 --> build_undirected_graph_4
    build_undirected_graph_4 --> build_undirected_graph_5
    build_undirected_graph_5 --> build_undirected_graph_6
    build_undirected_graph_6 --> build_undirected_graph_7
    build_undirected_graph_7 --> build_undirected_graph_8
    build_undirected_graph_8 --> build_undirected_graph_9
```

## Function: `collect_functions`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    collect_functions_0["ENTRY"]
    collect_functions_1["result . elements . iter () . filter (| elem | matches ! (elem . element_type..."]
    collect_functions_2["EXIT"]
    collect_functions_0 --> collect_functions_1
    collect_functions_1 --> collect_functions_2
```

## Function: `compute_cluster_cohesion`

- File: src/250_cohesion_analyzer.rs
- Branches: 1
- Loops: 0
- Nodes: 12
- Edges: 12

```mermaid
flowchart TD
    compute_cluster_cohesion_0["ENTRY"]
    compute_cluster_cohesion_1["let member_set : HashSet < usize > = members . iter () . copied () . collect ()"]
    compute_cluster_cohesion_2["let mut internal = 0usize"]
    compute_cluster_cohesion_3["let mut total = 0usize"]
    compute_cluster_cohesion_4["for idx in members { if let Some (outgoing) = outgoing_counts . get (* idx) {..."]
    compute_cluster_cohesion_5["if total == 0"]
    compute_cluster_cohesion_6["THEN BB"]
    compute_cluster_cohesion_7["0.0"]
    compute_cluster_cohesion_8["ELSE BB"]
    compute_cluster_cohesion_9["{ internal as f64 / total as f64 }"]
    compute_cluster_cohesion_10["IF JOIN"]
    compute_cluster_cohesion_11["EXIT"]
    compute_cluster_cohesion_0 --> compute_cluster_cohesion_1
    compute_cluster_cohesion_1 --> compute_cluster_cohesion_2
    compute_cluster_cohesion_2 --> compute_cluster_cohesion_3
    compute_cluster_cohesion_3 --> compute_cluster_cohesion_4
    compute_cluster_cohesion_4 --> compute_cluster_cohesion_5
    compute_cluster_cohesion_5 --> compute_cluster_cohesion_6
    compute_cluster_cohesion_6 --> compute_cluster_cohesion_7
    compute_cluster_cohesion_5 --> compute_cluster_cohesion_8
    compute_cluster_cohesion_8 --> compute_cluster_cohesion_9
    compute_cluster_cohesion_7 --> compute_cluster_cohesion_10
    compute_cluster_cohesion_9 --> compute_cluster_cohesion_10
    compute_cluster_cohesion_10 --> compute_cluster_cohesion_11
```

## Function: `compute_type_coupling`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    compute_type_coupling_0["ENTRY"]
    compute_type_coupling_1["let mut same_file = 0usize"]
    compute_type_coupling_2["let mut other_file = 0usize"]
    compute_type_coupling_3["let tokens = extract_identifiers (& func . signature)"]
    compute_type_coupling_4["let same_set = file_types . get (& func . file_path)"]
    compute_type_coupling_5["for token in tokens { if let Some (types) = same_set { if types . contains (&..."]
    compute_type_coupling_6["(same_file , other_file)"]
    compute_type_coupling_7["EXIT"]
    compute_type_coupling_0 --> compute_type_coupling_1
    compute_type_coupling_1 --> compute_type_coupling_2
    compute_type_coupling_2 --> compute_type_coupling_3
    compute_type_coupling_3 --> compute_type_coupling_4
    compute_type_coupling_4 --> compute_type_coupling_5
    compute_type_coupling_5 --> compute_type_coupling_6
    compute_type_coupling_6 --> compute_type_coupling_7
```

## Function: `determine_status`

- File: src/250_cohesion_analyzer.rs
- Branches: 4
- Loops: 0
- Nodes: 26
- Edges: 29

```mermaid
flowchart TD
    determine_status_0["ENTRY"]
    determine_status_1["let total_activity = call_analysis . intra_file_calls + call_analysis . calls_from_same_file"]
    determine_status_2["let external_activity : usize = call_analysis . calls_from_other_files . iter () . map (| (_ , count) | * cou..."]
    determine_status_3["if total_activity == 0 && external_activity == 0"]
    determine_status_4["THEN BB"]
    determine_status_5["return PlacementStatus :: Orphaned { suggested_module : 'utilities' . to_stri..."]
    determine_status_6["EMPTY ELSE"]
    determine_status_7["IF JOIN"]
    determine_status_8["if let Some ((current_layer , required_layer)) = layer_violation"]
    determine_status_9["THEN BB"]
    determine_status_10["return PlacementStatus :: LayerViolation { current_layer , required_layer , }"]
    determine_status_11["EMPTY ELSE"]
    determine_status_12["IF JOIN"]
    determine_status_13["if cohesion_score >= threshold"]
    determine_status_14["THEN BB"]
    determine_status_15["return PlacementStatus :: Correct"]
    determine_status_16["EMPTY ELSE"]
    determine_status_17["IF JOIN"]
    determine_status_18["let mut impact = threshold - cohesion_score"]
    determine_status_19["if impact < 0.0"]
    determine_status_20["THEN BB"]
    determine_status_21["impact = 0.0"]
    determine_status_22["EMPTY ELSE"]
    determine_status_23["IF JOIN"]
    determine_status_24["PlacementStatus :: ShouldMove { reason : format ! ('cohesion {:.2} below thre..."]
    determine_status_25["EXIT"]
    determine_status_0 --> determine_status_1
    determine_status_1 --> determine_status_2
    determine_status_2 --> determine_status_3
    determine_status_3 --> determine_status_4
    determine_status_4 --> determine_status_5
    determine_status_3 --> determine_status_6
    determine_status_5 --> determine_status_7
    determine_status_6 --> determine_status_7
    determine_status_7 --> determine_status_8
    determine_status_8 --> determine_status_9
    determine_status_9 --> determine_status_10
    determine_status_8 --> determine_status_11
    determine_status_10 --> determine_status_12
    determine_status_11 --> determine_status_12
    determine_status_12 --> determine_status_13
    determine_status_13 --> determine_status_14
    determine_status_14 --> determine_status_15
    determine_status_13 --> determine_status_16
    determine_status_15 --> determine_status_17
    determine_status_16 --> determine_status_17
    determine_status_17 --> determine_status_18
    determine_status_18 --> determine_status_19
    determine_status_19 --> determine_status_20
    determine_status_20 --> determine_status_21
    determine_status_19 --> determine_status_22
    determine_status_21 --> determine_status_23
    determine_status_22 --> determine_status_23
    determine_status_23 --> determine_status_24
    determine_status_24 --> determine_status_25
```

## Function: `extract_identifiers`

- File: src/250_cohesion_analyzer.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    extract_identifiers_0["ENTRY"]
    extract_identifiers_1["let mut tokens = Vec :: new ()"]
    extract_identifiers_2["let mut current = String :: new ()"]
    extract_identifiers_3["for ch in text . chars () { if ch . is_ascii_alphanumeric () || ch == '_' { c..."]
    extract_identifiers_4["if ! current . is_empty ()"]
    extract_identifiers_5["THEN BB"]
    extract_identifiers_6["tokens . push (current)"]
    extract_identifiers_7["EMPTY ELSE"]
    extract_identifiers_8["IF JOIN"]
    extract_identifiers_9["tokens"]
    extract_identifiers_10["EXIT"]
    extract_identifiers_0 --> extract_identifiers_1
    extract_identifiers_1 --> extract_identifiers_2
    extract_identifiers_2 --> extract_identifiers_3
    extract_identifiers_3 --> extract_identifiers_4
    extract_identifiers_4 --> extract_identifiers_5
    extract_identifiers_5 --> extract_identifiers_6
    extract_identifiers_4 --> extract_identifiers_7
    extract_identifiers_6 --> extract_identifiers_8
    extract_identifiers_7 --> extract_identifiers_8
    extract_identifiers_8 --> extract_identifiers_9
    extract_identifiers_9 --> extract_identifiers_10
```

## Function: `louvain_communities`

- File: src/250_cohesion_analyzer.rs
- Branches: 2
- Loops: 0
- Nodes: 20
- Edges: 21

```mermaid
flowchart TD
    louvain_communities_0["ENTRY"]
    louvain_communities_1["let n = outgoing_counts . len ()"]
    louvain_communities_2["if n == 0"]
    louvain_communities_3["THEN BB"]
    louvain_communities_4["return Vec :: new ()"]
    louvain_communities_5["EMPTY ELSE"]
    louvain_communities_6["IF JOIN"]
    louvain_communities_7["let (neighbors , degrees , total_weight) = build_undirected_graph (outgoing_counts)"]
    louvain_communities_8["if total_weight == 0"]
    louvain_communities_9["THEN BB"]
    louvain_communities_10["return (0 .. n) . collect ()"]
    louvain_communities_11["EMPTY ELSE"]
    louvain_communities_12["IF JOIN"]
    louvain_communities_13["let two_m = (2 * total_weight) as f64"]
    louvain_communities_14["let mut community : Vec < usize > = (0 .. n) . collect ()"]
    louvain_communities_15["let mut sum_tot = degrees . clone ()"]
    louvain_communities_16["let max_iters = 25"]
    louvain_communities_17["for iter in 0 .. max_iters { let mut moved = false ; for node in 0 .. n { let..."]
    louvain_communities_18["community"]
    louvain_communities_19["EXIT"]
    louvain_communities_0 --> louvain_communities_1
    louvain_communities_1 --> louvain_communities_2
    louvain_communities_2 --> louvain_communities_3
    louvain_communities_3 --> louvain_communities_4
    louvain_communities_2 --> louvain_communities_5
    louvain_communities_4 --> louvain_communities_6
    louvain_communities_5 --> louvain_communities_6
    louvain_communities_6 --> louvain_communities_7
    louvain_communities_7 --> louvain_communities_8
    louvain_communities_8 --> louvain_communities_9
    louvain_communities_9 --> louvain_communities_10
    louvain_communities_8 --> louvain_communities_11
    louvain_communities_10 --> louvain_communities_12
    louvain_communities_11 --> louvain_communities_12
    louvain_communities_12 --> louvain_communities_13
    louvain_communities_13 --> louvain_communities_14
    louvain_communities_14 --> louvain_communities_15
    louvain_communities_15 --> louvain_communities_16
    louvain_communities_16 --> louvain_communities_17
    louvain_communities_17 --> louvain_communities_18
    louvain_communities_18 --> louvain_communities_19
```

## Function: `suggest_cluster_file`

- File: src/250_cohesion_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    suggest_cluster_file_0["ENTRY"]
    suggest_cluster_file_1["let mut counts : HashMap < & str , usize > = HashMap :: new ()"]
    suggest_cluster_file_2["for idx in members { let path = functions [* idx] . file_path . as_str () ; *..."]
    suggest_cluster_file_3["counts . into_iter () . max_by_key (| (_ , count) | * count) . map (| (path ,..."]
    suggest_cluster_file_4["EXIT"]
    suggest_cluster_file_0 --> suggest_cluster_file_1
    suggest_cluster_file_1 --> suggest_cluster_file_2
    suggest_cluster_file_2 --> suggest_cluster_file_3
    suggest_cluster_file_3 --> suggest_cluster_file_4
```

## Function: `suggest_file`

- File: src/250_cohesion_analyzer.rs
- Branches: 3
- Loops: 0
- Nodes: 21
- Edges: 23

```mermaid
flowchart TD
    suggest_file_0["ENTRY"]
    suggest_file_1["let mut best_file : Option < PathBuf > = None"]
    suggest_file_2["let mut best_score = 0usize"]
    suggest_file_3["for (file , count) in & call_analysis . calls_from_other_files { if * count >..."]
    suggest_file_4["if best_score == 0"]
    suggest_file_5["THEN BB"]
    suggest_file_6["return None"]
    suggest_file_7["EMPTY ELSE"]
    suggest_file_8["IF JOIN"]
    suggest_file_9["if let Some (candidate) = & best_file"]
    suggest_file_10["THEN BB"]
    suggest_file_11["let outgoing_to_candidate = call_analysis . inter_file_calls . iter () . find (| (path , _) | path == can..."]
    suggest_file_12["if outgoing_to_candidate == 0 && call_analysis . intra_file_calls >= best_score"]
    suggest_file_13["THEN BB"]
    suggest_file_14["return None"]
    suggest_file_15["EMPTY ELSE"]
    suggest_file_16["IF JOIN"]
    suggest_file_17["EMPTY ELSE"]
    suggest_file_18["IF JOIN"]
    suggest_file_19["best_file"]
    suggest_file_20["EXIT"]
    suggest_file_0 --> suggest_file_1
    suggest_file_1 --> suggest_file_2
    suggest_file_2 --> suggest_file_3
    suggest_file_3 --> suggest_file_4
    suggest_file_4 --> suggest_file_5
    suggest_file_5 --> suggest_file_6
    suggest_file_4 --> suggest_file_7
    suggest_file_6 --> suggest_file_8
    suggest_file_7 --> suggest_file_8
    suggest_file_8 --> suggest_file_9
    suggest_file_9 --> suggest_file_10
    suggest_file_10 --> suggest_file_11
    suggest_file_11 --> suggest_file_12
    suggest_file_12 --> suggest_file_13
    suggest_file_13 --> suggest_file_14
    suggest_file_12 --> suggest_file_15
    suggest_file_14 --> suggest_file_16
    suggest_file_15 --> suggest_file_16
    suggest_file_9 --> suggest_file_17
    suggest_file_16 --> suggest_file_18
    suggest_file_17 --> suggest_file_18
    suggest_file_18 --> suggest_file_19
    suggest_file_19 --> suggest_file_20
```

