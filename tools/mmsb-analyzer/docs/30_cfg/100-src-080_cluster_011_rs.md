# CFG Group: src/080_cluster_011.rs

## Function: `build_directory_dag`

- File: src/080_cluster_011.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    build_directory_dag_0["ENTRY"]
    build_directory_dag_1["let files : Vec < PathBuf > = walkdir :: WalkDir :: new (dir) . into_iter () . filter_map (| e | e . ok ())..."]
    build_directory_dag_2["let file_set : HashSet < PathBuf > = files . iter () . cloned () . collect ()"]
    build_directory_dag_3["let module_map = build_module_map (& files)"]
    build_directory_dag_4["let dep_map = crate :: cluster_010 :: build_dependency_map (& files , & file_set , & module..."]
    build_directory_dag_5["let (graph , _) = build_file_dag (& files , & dep_map)"]
    build_directory_dag_6["Ok (graph)"]
    build_directory_dag_7["EXIT"]
    build_directory_dag_0 --> build_directory_dag_1
    build_directory_dag_1 --> build_directory_dag_2
    build_directory_dag_2 --> build_directory_dag_3
    build_directory_dag_3 --> build_directory_dag_4
    build_directory_dag_4 --> build_directory_dag_5
    build_directory_dag_5 --> build_directory_dag_6
    build_directory_dag_6 --> build_directory_dag_7
```

## Function: `build_file_dag`

- File: src/080_cluster_011.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    build_file_dag_0["ENTRY"]
    build_file_dag_1["let mut graph = DiGraph :: new ()"]
    build_file_dag_2["let mut node_map = HashMap :: new ()"]
    build_file_dag_3["for file in files { let node = graph . add_node (file . clone ()) ; node_map ..."]
    build_file_dag_4["for (file , deps) in dep_map { if let Some (& file_node) = node_map . get (fi..."]
    build_file_dag_5["(graph , node_map)"]
    build_file_dag_6["EXIT"]
    build_file_dag_0 --> build_file_dag_1
    build_file_dag_1 --> build_file_dag_2
    build_file_dag_2 --> build_file_dag_3
    build_file_dag_3 --> build_file_dag_4
    build_file_dag_4 --> build_file_dag_5
    build_file_dag_5 --> build_file_dag_6
```

## Function: `build_file_dependency_graph`

- File: src/080_cluster_011.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    build_file_dependency_graph_0["ENTRY"]
    build_file_dependency_graph_1["let file_set : HashSet < PathBuf > = files . iter () . cloned () . collect ()"]
    build_file_dependency_graph_2["let module_map = build_module_map (files)"]
    build_file_dependency_graph_3["let dep_map = crate :: cluster_010 :: build_dependency_map (files , & file_set , & module_m..."]
    build_file_dependency_graph_4["let (graph , _) = build_file_dag (files , & dep_map)"]
    build_file_dependency_graph_5["Ok (graph)"]
    build_file_dependency_graph_6["EXIT"]
    build_file_dependency_graph_0 --> build_file_dependency_graph_1
    build_file_dependency_graph_1 --> build_file_dependency_graph_2
    build_file_dependency_graph_2 --> build_file_dependency_graph_3
    build_file_dependency_graph_3 --> build_file_dependency_graph_4
    build_file_dependency_graph_4 --> build_file_dependency_graph_5
    build_file_dependency_graph_5 --> build_file_dependency_graph_6
```

## Function: `build_module_map`

- File: src/080_cluster_011.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    build_module_map_0["ENTRY"]
    build_module_map_1["let mut map = HashMap :: new ()"]
    build_module_map_2["for file in files { if let Some (stem) = file . file_stem () . and_then (| s ..."]
    build_module_map_3["map"]
    build_module_map_4["EXIT"]
    build_module_map_0 --> build_module_map_1
    build_module_map_1 --> build_module_map_2
    build_module_map_2 --> build_module_map_3
    build_module_map_3 --> build_module_map_4
```

## Function: `export_program_cfg_to_path`

- File: src/080_cluster_011.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    export_program_cfg_to_path_0["ENTRY"]
    export_program_cfg_to_path_1["use"]
    export_program_cfg_to_path_2["let mut program_cfg = ProgramCFG { functions : HashMap :: new () , call_edges : Vec :: new () , }"]
    export_program_cfg_to_path_3["for cfg in & result . cfgs { program_cfg . functions . insert (cfg . function..."]
    export_program_cfg_to_path_4["for (caller , callee) in call_edges { let caller_name = caller . split ('::')..."]
    export_program_cfg_to_path_5["let cfg_dir = output_path . join ('30_cfg')"]
    export_program_cfg_to_path_6["std :: fs :: create_dir_all (& cfg_dir) ?"]
    export_program_cfg_to_path_7["let dot_path = cfg_dir . join ('complete_program.dot')"]
    export_program_cfg_to_path_8["crate :: cluster_001 :: export_complete_program_dot (& program_cfg , dot_path..."]
    export_program_cfg_to_path_9["# [cfg (feature = 'png')] { let png_path = cfg_dir . join ('complete_program...."]
    export_program_cfg_to_path_10["Ok (())"]
    export_program_cfg_to_path_11["EXIT"]
    export_program_cfg_to_path_0 --> export_program_cfg_to_path_1
    export_program_cfg_to_path_1 --> export_program_cfg_to_path_2
    export_program_cfg_to_path_2 --> export_program_cfg_to_path_3
    export_program_cfg_to_path_3 --> export_program_cfg_to_path_4
    export_program_cfg_to_path_4 --> export_program_cfg_to_path_5
    export_program_cfg_to_path_5 --> export_program_cfg_to_path_6
    export_program_cfg_to_path_6 --> export_program_cfg_to_path_7
    export_program_cfg_to_path_7 --> export_program_cfg_to_path_8
    export_program_cfg_to_path_8 --> export_program_cfg_to_path_9
    export_program_cfg_to_path_9 --> export_program_cfg_to_path_10
    export_program_cfg_to_path_10 --> export_program_cfg_to_path_11
```

## Function: `resolve_path`

- File: src/080_cluster_011.rs
- Branches: 3
- Loops: 0
- Nodes: 18
- Edges: 20

```mermaid
flowchart TD
    resolve_path_0["ENTRY"]
    resolve_path_1["if file_set . contains (candidate)"]
    resolve_path_2["THEN BB"]
    resolve_path_3["return Some (candidate . to_path_buf ())"]
    resolve_path_4["EMPTY ELSE"]
    resolve_path_5["IF JOIN"]
    resolve_path_6["if let Some (file_name) = candidate . file_stem () . and_then (| s | s . to_s..."]
    resolve_path_7["THEN BB"]
    resolve_path_8["let key = crate :: cluster_010 :: normalize_module_name (file_name)"]
    resolve_path_9["if let Some (path) = module_map . get (& key)"]
    resolve_path_10["THEN BB"]
    resolve_path_11["return Some (path . clone ())"]
    resolve_path_12["EMPTY ELSE"]
    resolve_path_13["IF JOIN"]
    resolve_path_14["EMPTY ELSE"]
    resolve_path_15["IF JOIN"]
    resolve_path_16["None"]
    resolve_path_17["EXIT"]
    resolve_path_0 --> resolve_path_1
    resolve_path_1 --> resolve_path_2
    resolve_path_2 --> resolve_path_3
    resolve_path_1 --> resolve_path_4
    resolve_path_3 --> resolve_path_5
    resolve_path_4 --> resolve_path_5
    resolve_path_5 --> resolve_path_6
    resolve_path_6 --> resolve_path_7
    resolve_path_7 --> resolve_path_8
    resolve_path_8 --> resolve_path_9
    resolve_path_9 --> resolve_path_10
    resolve_path_10 --> resolve_path_11
    resolve_path_9 --> resolve_path_12
    resolve_path_11 --> resolve_path_13
    resolve_path_12 --> resolve_path_13
    resolve_path_6 --> resolve_path_14
    resolve_path_13 --> resolve_path_15
    resolve_path_14 --> resolve_path_15
    resolve_path_15 --> resolve_path_16
    resolve_path_16 --> resolve_path_17
```

