# CFG Group: src/000_dependency_analysis.rs

## Function: `analyze_file_ordering`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    analyze_file_ordering_0["ENTRY"]
    analyze_file_ordering_1["let step = step . unwrap_or (10)"]
    analyze_file_ordering_2["let file_set : HashSet < PathBuf > = files . iter () . cloned () . collect ()"]
    analyze_file_ordering_3["let module_map = crate :: cluster_011 :: build_module_map (files)"]
    analyze_file_ordering_4["let dep_map = crate :: cluster_010 :: build_dependency_map (files , & file_set , & module_m..."]
    analyze_file_ordering_5["let file_layers = build_file_layers (files)"]
    analyze_file_ordering_6["let ordered_directories = crate :: layer_core :: order_directories (files , & dep_map)"]
    analyze_file_ordering_7["let (graph , node_map) = crate :: cluster_011 :: build_file_dag (files , & dep_map)"]
    analyze_file_ordering_8["let layer_violations = crate :: cluster_008 :: detect_layer_violations (& graph , & file_layers)"]
    analyze_file_ordering_9["let cycles = detect_cycles (& graph , files)"]
    analyze_file_ordering_10["let ordered_nodes = if cycles . is_empty () { crate :: layer_core :: layer_constrained_sort (& gr..."]
    analyze_file_ordering_11["let ordered_files = ordered_nodes . into_iter () . map (| idx | graph [idx] . clone ()) . collect..."]
    analyze_file_ordering_12["let file_entries = build_entries (& ordered_files , step)"]
    analyze_file_ordering_13["let violations = detect_violations (& file_entries , & dep_map)"]
    analyze_file_ordering_14["Ok (crate :: types :: FileOrderingResult { ordered_files : file_entries , vio..."]
    analyze_file_ordering_15["EXIT"]
    analyze_file_ordering_0 --> analyze_file_ordering_1
    analyze_file_ordering_1 --> analyze_file_ordering_2
    analyze_file_ordering_2 --> analyze_file_ordering_3
    analyze_file_ordering_3 --> analyze_file_ordering_4
    analyze_file_ordering_4 --> analyze_file_ordering_5
    analyze_file_ordering_5 --> analyze_file_ordering_6
    analyze_file_ordering_6 --> analyze_file_ordering_7
    analyze_file_ordering_7 --> analyze_file_ordering_8
    analyze_file_ordering_8 --> analyze_file_ordering_9
    analyze_file_ordering_9 --> analyze_file_ordering_10
    analyze_file_ordering_10 --> analyze_file_ordering_11
    analyze_file_ordering_11 --> analyze_file_ordering_12
    analyze_file_ordering_12 --> analyze_file_ordering_13
    analyze_file_ordering_13 --> analyze_file_ordering_14
    analyze_file_ordering_14 --> analyze_file_ordering_15
```

## Function: `build_directory_entry_map`

- File: src/000_dependency_analysis.rs
- Branches: 1
- Loops: 0
- Nodes: 25
- Edges: 25

```mermaid
flowchart TD
    build_directory_entry_map_0["ENTRY"]
    build_directory_entry_map_1["use"]
    build_directory_entry_map_2["use"]
    build_directory_entry_map_3["use"]
    build_directory_entry_map_4["use"]
    build_directory_entry_map_5["use"]
    build_directory_entry_map_6["item"]
    build_directory_entry_map_7["if files . is_empty ()"]
    build_directory_entry_map_8["THEN BB"]
    build_directory_entry_map_9["return Ok (HashMap :: new ())"]
    build_directory_entry_map_10["EMPTY ELSE"]
    build_directory_entry_map_11["IF JOIN"]
    build_directory_entry_map_12["let file_set : HashSet < PathBuf > = files . iter () . cloned () . collect ()"]
    build_directory_entry_map_13["let module_map = crate :: cluster_011 :: build_module_map (files)"]
    build_directory_entry_map_14["let dep_map = build_dependency_map (files , & file_set , & module_map) ?"]
    build_directory_entry_map_15["let file_layers = build_file_layers (files)"]
    build_directory_entry_map_16["let (graph , node_map) = build_file_dag (files , & dep_map)"]
    build_directory_entry_map_17["let cycles = detect_cycles (& graph , files)"]
    build_directory_entry_map_18["let ordered_nodes = if cycles . is_empty () { layer_constrained_sort (& graph , & file_layers) . ..."]
    build_directory_entry_map_19["let ordered_files = ordered_nodes . into_iter () . map (| idx | graph [idx] . clone ()) . collect..."]
    build_directory_entry_map_20["let ordering = FileOrderingResult { ordered_files : build_entries (& ordered_files , DEFAULT..."]
    build_directory_entry_map_21["let mut map = HashMap :: new ()"]
    build_directory_entry_map_22["for entry in ordering . ordered_files { map . insert (entry . current_path . ..."]
    build_directory_entry_map_23["Ok (map)"]
    build_directory_entry_map_24["EXIT"]
    build_directory_entry_map_0 --> build_directory_entry_map_1
    build_directory_entry_map_1 --> build_directory_entry_map_2
    build_directory_entry_map_2 --> build_directory_entry_map_3
    build_directory_entry_map_3 --> build_directory_entry_map_4
    build_directory_entry_map_4 --> build_directory_entry_map_5
    build_directory_entry_map_5 --> build_directory_entry_map_6
    build_directory_entry_map_6 --> build_directory_entry_map_7
    build_directory_entry_map_7 --> build_directory_entry_map_8
    build_directory_entry_map_8 --> build_directory_entry_map_9
    build_directory_entry_map_7 --> build_directory_entry_map_10
    build_directory_entry_map_9 --> build_directory_entry_map_11
    build_directory_entry_map_10 --> build_directory_entry_map_11
    build_directory_entry_map_11 --> build_directory_entry_map_12
    build_directory_entry_map_12 --> build_directory_entry_map_13
    build_directory_entry_map_13 --> build_directory_entry_map_14
    build_directory_entry_map_14 --> build_directory_entry_map_15
    build_directory_entry_map_15 --> build_directory_entry_map_16
    build_directory_entry_map_16 --> build_directory_entry_map_17
    build_directory_entry_map_17 --> build_directory_entry_map_18
    build_directory_entry_map_18 --> build_directory_entry_map_19
    build_directory_entry_map_19 --> build_directory_entry_map_20
    build_directory_entry_map_20 --> build_directory_entry_map_21
    build_directory_entry_map_21 --> build_directory_entry_map_22
    build_directory_entry_map_22 --> build_directory_entry_map_23
    build_directory_entry_map_23 --> build_directory_entry_map_24
```

## Function: `build_entries`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    build_entries_0["ENTRY"]
    build_entries_1["ordered . iter () . enumerate () . map (| (idx , path) | { let canonical_orde..."]
    build_entries_2["EXIT"]
    build_entries_0 --> build_entries_1
    build_entries_1 --> build_entries_2
```

## Function: `build_file_layers`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    build_file_layers_0["ENTRY"]
    build_file_layers_1["let mut layers = HashMap :: new ()"]
    build_file_layers_2["for file in files { layers . insert (file . clone () , detect_layer (file)) ; }"]
    build_file_layers_3["layers"]
    build_file_layers_4["EXIT"]
    build_file_layers_0 --> build_file_layers_1
    build_file_layers_1 --> build_file_layers_2
    build_file_layers_2 --> build_file_layers_3
    build_file_layers_3 --> build_file_layers_4
```

## Function: `collect_julia_dependencies`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    collect_julia_dependencies_0["ENTRY"]
    collect_julia_dependencies_1["let content = fs :: read_to_string (path) . with_context (| | format ! ('Unable to read Jul..."]
    collect_julia_dependencies_2["let mut deps = Vec :: new ()"]
    collect_julia_dependencies_3["for cap in INCLUDE_REGEX . captures_iter (& content) { if let Some (path_matc..."]
    collect_julia_dependencies_4["for cap in USING_REGEX . captures_iter (& content) { if let Some (module_matc..."]
    collect_julia_dependencies_5["for cap in ROOT_USING_REGEX . captures_iter (& content) { if let Some (symbol..."]
    collect_julia_dependencies_6["for cap in LOCAL_USING_REGEX . captures_iter (& content) { if let Some (modul..."]
    collect_julia_dependencies_7["Ok (deps)"]
    collect_julia_dependencies_8["EXIT"]
    collect_julia_dependencies_0 --> collect_julia_dependencies_1
    collect_julia_dependencies_1 --> collect_julia_dependencies_2
    collect_julia_dependencies_2 --> collect_julia_dependencies_3
    collect_julia_dependencies_3 --> collect_julia_dependencies_4
    collect_julia_dependencies_4 --> collect_julia_dependencies_5
    collect_julia_dependencies_5 --> collect_julia_dependencies_6
    collect_julia_dependencies_6 --> collect_julia_dependencies_7
    collect_julia_dependencies_7 --> collect_julia_dependencies_8
```

## Function: `collect_naming_warnings`

- File: src/000_dependency_analysis.rs
- Branches: 1
- Loops: 0
- Nodes: 13
- Edges: 13

```mermaid
flowchart TD
    collect_naming_warnings_0["ENTRY"]
    collect_naming_warnings_1["use"]
    collect_naming_warnings_2["use"]
    collect_naming_warnings_3["if directory . path . components () . any (| comp | comp . as_os_str () == '_..."]
    collect_naming_warnings_4["THEN BB"]
    collect_naming_warnings_5["return Ok (())"]
    collect_naming_warnings_6["EMPTY ELSE"]
    collect_naming_warnings_7["IF JOIN"]
    collect_naming_warnings_8["let file_map = build_directory_entry_map (& directory . files) ?"]
    collect_naming_warnings_9["for file in & directory . files { if file . components () . any (| comp | com..."]
    collect_naming_warnings_10["for child in & directory . subdirectories { collect_naming_warnings (child , ..."]
    collect_naming_warnings_11["Ok (())"]
    collect_naming_warnings_12["EXIT"]
    collect_naming_warnings_0 --> collect_naming_warnings_1
    collect_naming_warnings_1 --> collect_naming_warnings_2
    collect_naming_warnings_2 --> collect_naming_warnings_3
    collect_naming_warnings_3 --> collect_naming_warnings_4
    collect_naming_warnings_4 --> collect_naming_warnings_5
    collect_naming_warnings_3 --> collect_naming_warnings_6
    collect_naming_warnings_5 --> collect_naming_warnings_7
    collect_naming_warnings_6 --> collect_naming_warnings_7
    collect_naming_warnings_7 --> collect_naming_warnings_8
    collect_naming_warnings_8 --> collect_naming_warnings_9
    collect_naming_warnings_9 --> collect_naming_warnings_10
    collect_naming_warnings_10 --> collect_naming_warnings_11
    collect_naming_warnings_11 --> collect_naming_warnings_12
```

## Function: `collect_roots_from_crate`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    collect_roots_from_crate_0["ENTRY"]
    collect_roots_from_crate_1["match tree { UseTree :: Path (path) => { let ident = path . ident . to_string..."]
    collect_roots_from_crate_2["EXIT"]
    collect_roots_from_crate_0 --> collect_roots_from_crate_1
    collect_roots_from_crate_1 --> collect_roots_from_crate_2
```

## Function: `collect_rust_dependencies`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    collect_rust_dependencies_0["ENTRY"]
    collect_rust_dependencies_1["let content = fs :: read_to_string (path) . with_context (| | format ! ('Unable to read Rus..."]
    collect_rust_dependencies_2["let syntax = syn :: parse_file (& content) . with_context (| | format ! ('Unable to parse ..."]
    collect_rust_dependencies_3["let mut collector = UseCollector :: default ()"]
    collect_rust_dependencies_4["collector . visit_file (& syntax)"]
    collect_rust_dependencies_5["Ok (collector . deps)"]
    collect_rust_dependencies_6["EXIT"]
    collect_rust_dependencies_0 --> collect_rust_dependencies_1
    collect_rust_dependencies_1 --> collect_rust_dependencies_2
    collect_rust_dependencies_2 --> collect_rust_dependencies_3
    collect_rust_dependencies_3 --> collect_rust_dependencies_4
    collect_rust_dependencies_4 --> collect_rust_dependencies_5
    collect_rust_dependencies_5 --> collect_rust_dependencies_6
```

## Function: `detect_cycles`

- File: src/000_dependency_analysis.rs
- Branches: 2
- Loops: 0
- Nodes: 18
- Edges: 19

```mermaid
flowchart TD
    detect_cycles_0["ENTRY"]
    detect_cycles_1["let sccs = tarjan_scc (graph)"]
    detect_cycles_2["let mut cycles = Vec :: new ()"]
    detect_cycles_3["for scc in sccs { if scc . len () > 1 { cycles . push (scc . into_iter () . m..."]
    detect_cycles_4["if cycles . is_empty ()"]
    detect_cycles_5["THEN BB"]
    detect_cycles_6["return cycles"]
    detect_cycles_7["EMPTY ELSE"]
    detect_cycles_8["IF JOIN"]
    detect_cycles_9["if cycles . iter () . all (| cycle | cycle . is_empty ())"]
    detect_cycles_10["THEN BB"]
    detect_cycles_11["let mut fallback = files . to_vec ()"]
    detect_cycles_12["fallback . sort ()"]
    detect_cycles_13["cycles . push (fallback)"]
    detect_cycles_14["EMPTY ELSE"]
    detect_cycles_15["IF JOIN"]
    detect_cycles_16["cycles"]
    detect_cycles_17["EXIT"]
    detect_cycles_0 --> detect_cycles_1
    detect_cycles_1 --> detect_cycles_2
    detect_cycles_2 --> detect_cycles_3
    detect_cycles_3 --> detect_cycles_4
    detect_cycles_4 --> detect_cycles_5
    detect_cycles_5 --> detect_cycles_6
    detect_cycles_4 --> detect_cycles_7
    detect_cycles_6 --> detect_cycles_8
    detect_cycles_7 --> detect_cycles_8
    detect_cycles_8 --> detect_cycles_9
    detect_cycles_9 --> detect_cycles_10
    detect_cycles_10 --> detect_cycles_11
    detect_cycles_11 --> detect_cycles_12
    detect_cycles_12 --> detect_cycles_13
    detect_cycles_9 --> detect_cycles_14
    detect_cycles_13 --> detect_cycles_15
    detect_cycles_14 --> detect_cycles_15
    detect_cycles_15 --> detect_cycles_16
    detect_cycles_16 --> detect_cycles_17
```

## Function: `detect_layer`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    detect_layer_0["ENTRY"]
    detect_layer_1["for component in path . components () { if let Some (name) = component . as_o..."]
    detect_layer_2["'root' . to_string ()"]
    detect_layer_3["EXIT"]
    detect_layer_0 --> detect_layer_1
    detect_layer_1 --> detect_layer_2
    detect_layer_2 --> detect_layer_3
```

## Function: `detect_violations`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    detect_violations_0["ENTRY"]
    detect_violations_1["let mut alpha = ordered_files . to_vec ()"]
    detect_violations_2["alpha . sort_by (| a , b | a . current_path . cmp (& b . current_path))"]
    detect_violations_3["let alpha_positions : HashMap < PathBuf , usize > = alpha . iter () . enumerate () . map (| (idx , entry) | (entry . current_path..."]
    detect_violations_4["let canonical_positions : HashMap < PathBuf , usize > = ordered_files . iter () . enumerate () . map (| (idx , entry) | (entry . curr..."]
    detect_violations_5["let mut violations = Vec :: new ()"]
    detect_violations_6["for entry in ordered_files { let Some (& alpha_pos) = alpha_positions . get (..."]
    detect_violations_7["violations"]
    detect_violations_8["EXIT"]
    detect_violations_0 --> detect_violations_1
    detect_violations_1 --> detect_violations_2
    detect_violations_2 --> detect_violations_3
    detect_violations_3 --> detect_violations_4
    detect_violations_4 --> detect_violations_5
    detect_violations_5 --> detect_violations_6
    detect_violations_6 --> detect_violations_7
    detect_violations_7 --> detect_violations_8
```

## Function: `detects_cycles`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    detects_cycles_0["ENTRY"]
    detects_cycles_1["use"]
    detects_cycles_2["use"]
    detects_cycles_3["let dir = temp_dir ('cycle')"]
    detects_cycles_4["create_dir_all (& dir) ?"]
    detects_cycles_5["let a = dir . join ('a.rs')"]
    detects_cycles_6["let b = dir . join ('b.rs')"]
    detects_cycles_7["write (& a , 'use crate::b; pub fn a() {}') ?"]
    detects_cycles_8["write (& b , 'use crate::a; pub fn b() {}') ?"]
    detects_cycles_9["let result = analyze_file_ordering (& [a . clone () , b . clone ()] , Some (10)) ?"]
    detects_cycles_10["macro assert"]
    detects_cycles_11["Ok (())"]
    detects_cycles_12["EXIT"]
    detects_cycles_0 --> detects_cycles_1
    detects_cycles_1 --> detects_cycles_2
    detects_cycles_2 --> detects_cycles_3
    detects_cycles_3 --> detects_cycles_4
    detects_cycles_4 --> detects_cycles_5
    detects_cycles_5 --> detects_cycles_6
    detects_cycles_6 --> detects_cycles_7
    detects_cycles_7 --> detects_cycles_8
    detects_cycles_8 --> detects_cycles_9
    detects_cycles_9 --> detects_cycles_10
    detects_cycles_10 --> detects_cycles_11
    detects_cycles_11 --> detects_cycles_12
```

## Function: `escape_dot`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    escape_dot_0["ENTRY"]
    escape_dot_1["s . replace ('\\' , '\\\\') . replace (''' , '\\\'') . replace ('\n' , '\\n')"]
    escape_dot_2["EXIT"]
    escape_dot_0 --> escape_dot_1
    escape_dot_1 --> escape_dot_2
```

## Function: `export_complete_program_dot`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 26
- Edges: 25

```mermaid
flowchart TD
    export_complete_program_dot_0["ENTRY"]
    export_complete_program_dot_1["use"]
    export_complete_program_dot_2["use"]
    export_complete_program_dot_3["item"]
    export_complete_program_dot_4["let mut dot = String :: new ()"]
    export_complete_program_dot_5["writeln ! (dot , 'digraph ProgramCFG {{') . unwrap ()"]
    export_complete_program_dot_6["writeln ! (dot , ' rankdir=TB;') . unwrap ()"]
    export_complete_program_dot_7["writeln ! (dot , ' compound=true;') . unwrap ()"]
    export_complete_program_dot_8["writeln ! (dot , ' newrank=true;') . unwrap ()"]
    export_complete_program_dot_9["writeln ! (dot , ' label=\'Complete Program CFG - {} functions\';' , program ..."]
    export_complete_program_dot_10["writeln ! (dot , ' labelloc=t;') . unwrap ()"]
    export_complete_program_dot_11["writeln ! (dot , ' fontsize=16;') . unwrap ()"]
    export_complete_program_dot_12["writeln ! (dot , '') . unwrap ()"]
    export_complete_program_dot_13["let mut funcs : Vec < _ > = program . functions . iter () . collect ()"]
    export_complete_program_dot_14["funcs . sort_by_key (| (fid , _) | fid . as_str ())"]
    export_complete_program_dot_15["let mut func_to_cluster : HashMap < & String , usize > = HashMap :: new ()"]
    export_complete_program_dot_16["for (cluster_idx , (func_id , cfg)) in funcs . iter () . enumerate () { let s..."]
    export_complete_program_dot_17["writeln ! (dot , ' // Inter-function calls') . unwrap ()"]
    export_complete_program_dot_18["writeln ! (dot , ' edge [style=dashed, color=blue, penwidth=2];') . unwrap ()"]
    export_complete_program_dot_19["writeln ! (dot , '') . unwrap ()"]
    export_complete_program_dot_20["for (caller , callee) in & program . call_edges { if let (Some (& caller_idx)..."]
    export_complete_program_dot_21["writeln ! (dot , '}}') . unwrap ()"]
    export_complete_program_dot_22["std :: fs :: write (path , dot) ?"]
    export_complete_program_dot_23["macro println"]
    export_complete_program_dot_24["Ok (())"]
    export_complete_program_dot_25["EXIT"]
    export_complete_program_dot_0 --> export_complete_program_dot_1
    export_complete_program_dot_1 --> export_complete_program_dot_2
    export_complete_program_dot_2 --> export_complete_program_dot_3
    export_complete_program_dot_3 --> export_complete_program_dot_4
    export_complete_program_dot_4 --> export_complete_program_dot_5
    export_complete_program_dot_5 --> export_complete_program_dot_6
    export_complete_program_dot_6 --> export_complete_program_dot_7
    export_complete_program_dot_7 --> export_complete_program_dot_8
    export_complete_program_dot_8 --> export_complete_program_dot_9
    export_complete_program_dot_9 --> export_complete_program_dot_10
    export_complete_program_dot_10 --> export_complete_program_dot_11
    export_complete_program_dot_11 --> export_complete_program_dot_12
    export_complete_program_dot_12 --> export_complete_program_dot_13
    export_complete_program_dot_13 --> export_complete_program_dot_14
    export_complete_program_dot_14 --> export_complete_program_dot_15
    export_complete_program_dot_15 --> export_complete_program_dot_16
    export_complete_program_dot_16 --> export_complete_program_dot_17
    export_complete_program_dot_17 --> export_complete_program_dot_18
    export_complete_program_dot_18 --> export_complete_program_dot_19
    export_complete_program_dot_19 --> export_complete_program_dot_20
    export_complete_program_dot_20 --> export_complete_program_dot_21
    export_complete_program_dot_21 --> export_complete_program_dot_22
    export_complete_program_dot_22 --> export_complete_program_dot_23
    export_complete_program_dot_23 --> export_complete_program_dot_24
    export_complete_program_dot_24 --> export_complete_program_dot_25
```

## Function: `gather_julia_files`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    gather_julia_files_0["ENTRY"]
    gather_julia_files_1["use"]
    gather_julia_files_2["let src_root = crate :: layer_utilities :: resolve_source_root (root)"]
    gather_julia_files_3["WalkDir :: new (& src_root) . into_iter () . filter_entry (| entry | { if ent..."]
    gather_julia_files_4["EXIT"]
    gather_julia_files_0 --> gather_julia_files_1
    gather_julia_files_1 --> gather_julia_files_2
    gather_julia_files_2 --> gather_julia_files_3
    gather_julia_files_3 --> gather_julia_files_4
```

## Function: `generates_canonical_names_and_violations`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    generates_canonical_names_and_violations_0["ENTRY"]
    generates_canonical_names_and_violations_1["use"]
    generates_canonical_names_and_violations_2["use"]
    generates_canonical_names_and_violations_3["let dir = temp_dir ('names')"]
    generates_canonical_names_and_violations_4["create_dir_all (& dir) ?"]
    generates_canonical_names_and_violations_5["let a = dir . join ('a.rs')"]
    generates_canonical_names_and_violations_6["let b = dir . join ('b.rs')"]
    generates_canonical_names_and_violations_7["write (& a , 'use crate::b; pub fn a() {}') ?"]
    generates_canonical_names_and_violations_8["write (& b , 'pub fn b() {}') ?"]
    generates_canonical_names_and_violations_9["let result = analyze_file_ordering (& [a . clone () , b . clone ()] , Some (10)) ?"]
    generates_canonical_names_and_violations_10["let entries = & result . ordered_files"]
    generates_canonical_names_and_violations_11["macro assert_eq"]
    generates_canonical_names_and_violations_12["macro assert_eq"]
    generates_canonical_names_and_violations_13["macro assert"]
    generates_canonical_names_and_violations_14["Ok (())"]
    generates_canonical_names_and_violations_15["EXIT"]
    generates_canonical_names_and_violations_0 --> generates_canonical_names_and_violations_1
    generates_canonical_names_and_violations_1 --> generates_canonical_names_and_violations_2
    generates_canonical_names_and_violations_2 --> generates_canonical_names_and_violations_3
    generates_canonical_names_and_violations_3 --> generates_canonical_names_and_violations_4
    generates_canonical_names_and_violations_4 --> generates_canonical_names_and_violations_5
    generates_canonical_names_and_violations_5 --> generates_canonical_names_and_violations_6
    generates_canonical_names_and_violations_6 --> generates_canonical_names_and_violations_7
    generates_canonical_names_and_violations_7 --> generates_canonical_names_and_violations_8
    generates_canonical_names_and_violations_8 --> generates_canonical_names_and_violations_9
    generates_canonical_names_and_violations_9 --> generates_canonical_names_and_violations_10
    generates_canonical_names_and_violations_10 --> generates_canonical_names_and_violations_11
    generates_canonical_names_and_violations_11 --> generates_canonical_names_and_violations_12
    generates_canonical_names_and_violations_12 --> generates_canonical_names_and_violations_13
    generates_canonical_names_and_violations_13 --> generates_canonical_names_and_violations_14
    generates_canonical_names_and_violations_14 --> generates_canonical_names_and_violations_15
```

## Function: `julia_entry_paths`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    julia_entry_paths_0["ENTRY"]
    julia_entry_paths_1["let src_dir = crate :: layer_utilities :: resolve_source_root (root)"]
    julia_entry_paths_2["['MMSB.jl' , 'API.jl' , 'MMSB/API.jl'] . iter () . map (| rel | src_dir . joi..."]
    julia_entry_paths_3["EXIT"]
    julia_entry_paths_0 --> julia_entry_paths_1
    julia_entry_paths_1 --> julia_entry_paths_2
    julia_entry_paths_2 --> julia_entry_paths_3
```

## Function: `layer_constrained_sort`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    layer_constrained_sort_0["ENTRY"]
    layer_constrained_sort_1["use"]
    layer_constrained_sort_2["let mut layer_nodes : BTreeMap < i32 , Vec < NodeIndex > > = BTreeMap :: new ()"]
    layer_constrained_sort_3["for node in graph . node_indices () { let file = & graph [node] ; let layer_n..."]
    layer_constrained_sort_4["let mut ordered = Vec :: new ()"]
    layer_constrained_sort_5["for (_layer , nodes) in layer_nodes { let sorted = topo_sort_within (graph , ..."]
    layer_constrained_sort_6["Ok (ordered)"]
    layer_constrained_sort_7["EXIT"]
    layer_constrained_sort_0 --> layer_constrained_sort_1
    layer_constrained_sort_1 --> layer_constrained_sort_2
    layer_constrained_sort_2 --> layer_constrained_sort_3
    layer_constrained_sort_3 --> layer_constrained_sort_4
    layer_constrained_sort_4 --> layer_constrained_sort_5
    layer_constrained_sort_5 --> layer_constrained_sort_6
    layer_constrained_sort_6 --> layer_constrained_sort_7
```

## Function: `naming_score_for_file`

- File: src/000_dependency_analysis.rs
- Branches: 12
- Loops: 0
- Nodes: 75
- Edges: 86

```mermaid
flowchart TD
    naming_score_for_file_0["ENTRY"]
    naming_score_for_file_1["let name = file . file_name () ? . to_string_lossy ()"]
    naming_score_for_file_2["let stem = file . file_stem () ? . to_string_lossy ()"]
    naming_score_for_file_3["let mut score = 1.0f64"]
    naming_score_for_file_4["if stem . len () < 3"]
    naming_score_for_file_5["THEN BB"]
    naming_score_for_file_6["score -= 0.2"]
    naming_score_for_file_7["EMPTY ELSE"]
    naming_score_for_file_8["IF JOIN"]
    naming_score_for_file_9["if stem . len () > 40"]
    naming_score_for_file_10["THEN BB"]
    naming_score_for_file_11["score -= 0.1"]
    naming_score_for_file_12["EMPTY ELSE"]
    naming_score_for_file_13["IF JOIN"]
    naming_score_for_file_14["if stem . chars () . any (| c | c . is_uppercase ())"]
    naming_score_for_file_15["THEN BB"]
    naming_score_for_file_16["score -= 0.1"]
    naming_score_for_file_17["EMPTY ELSE"]
    naming_score_for_file_18["IF JOIN"]
    naming_score_for_file_19["if ! stem . chars () . all (| c | c . is_ascii_lowercase () || c . is_ascii_d..."]
    naming_score_for_file_20["THEN BB"]
    naming_score_for_file_21["score -= 0.1"]
    naming_score_for_file_22["EMPTY ELSE"]
    naming_score_for_file_23["IF JOIN"]
    naming_score_for_file_24["if name . contains ('__')"]
    naming_score_for_file_25["THEN BB"]
    naming_score_for_file_26["score -= 0.1"]
    naming_score_for_file_27["EMPTY ELSE"]
    naming_score_for_file_28["IF JOIN"]
    naming_score_for_file_29["if let Some (entry) = order_entry"]
    naming_score_for_file_30["THEN BB"]
    naming_score_for_file_31["let expected = entry . suggested_name . as_str ()"]
    naming_score_for_file_32["let actual = name . as_ref ()"]
    naming_score_for_file_33["if expected != actual"]
    naming_score_for_file_34["THEN BB"]
    naming_score_for_file_35["score -= 0.3"]
    naming_score_for_file_36["ELSE BB"]
    naming_score_for_file_37["{ score += 0.1 ; }"]
    naming_score_for_file_38["IF JOIN"]
    naming_score_for_file_39["EMPTY ELSE"]
    naming_score_for_file_40["IF JOIN"]
    naming_score_for_file_41["if let Ok (contents) = fs :: read_to_string (file)"]
    naming_score_for_file_42["THEN BB"]
    naming_score_for_file_43["let mut ident_counts : HashMap < String , usize > = HashMap :: new ()"]
    naming_score_for_file_44["let ident_re = match Regex :: new (r'[A-Za-z_][A-Za-z0-9_]*') { Ok (regex) => regex , Err (_..."]
    naming_score_for_file_45["for cap in ident_re . captures_iter (& contents) { let Some (m) = cap . get (..."]
    naming_score_for_file_46["let mut idents = ident_counts . into_iter () . collect :: < Vec < _ > > ()"]
    naming_score_for_file_47["idents . sort_by (| a , b | b . 1 . cmp (& a . 1))"]
    naming_score_for_file_48["let top_idents = idents . into_iter () . take (8) . map (| (k , _) | k) . collect :: < Vec < _..."]
    naming_score_for_file_49["let name_tokens = stem . split ('_') . map (| s | s . to_lowercase ()) . filter (| s | ! s . is..."]
    naming_score_for_file_50["let overlap = top_idents . iter () . filter (| ident | name_tokens . iter () . any (| t | t..."]
    naming_score_for_file_51["if overlap == 0"]
    naming_score_for_file_52["THEN BB"]
    naming_score_for_file_53["score -= 0.1"]
    naming_score_for_file_54["ELSE BB"]
    naming_score_for_file_55["if overlap >= 2"]
    naming_score_for_file_56["THEN BB"]
    naming_score_for_file_57["score += 0.1"]
    naming_score_for_file_58["EMPTY ELSE"]
    naming_score_for_file_59["IF JOIN"]
    naming_score_for_file_60["IF JOIN"]
    naming_score_for_file_61["EMPTY ELSE"]
    naming_score_for_file_62["IF JOIN"]
    naming_score_for_file_63["if score < 0.0"]
    naming_score_for_file_64["THEN BB"]
    naming_score_for_file_65["score = 0.0"]
    naming_score_for_file_66["EMPTY ELSE"]
    naming_score_for_file_67["IF JOIN"]
    naming_score_for_file_68["if score > 1.0"]
    naming_score_for_file_69["THEN BB"]
    naming_score_for_file_70["score = 1.0"]
    naming_score_for_file_71["EMPTY ELSE"]
    naming_score_for_file_72["IF JOIN"]
    naming_score_for_file_73["Some (score * 100.0)"]
    naming_score_for_file_74["EXIT"]
    naming_score_for_file_0 --> naming_score_for_file_1
    naming_score_for_file_1 --> naming_score_for_file_2
    naming_score_for_file_2 --> naming_score_for_file_3
    naming_score_for_file_3 --> naming_score_for_file_4
    naming_score_for_file_4 --> naming_score_for_file_5
    naming_score_for_file_5 --> naming_score_for_file_6
    naming_score_for_file_4 --> naming_score_for_file_7
    naming_score_for_file_6 --> naming_score_for_file_8
    naming_score_for_file_7 --> naming_score_for_file_8
    naming_score_for_file_8 --> naming_score_for_file_9
    naming_score_for_file_9 --> naming_score_for_file_10
    naming_score_for_file_10 --> naming_score_for_file_11
    naming_score_for_file_9 --> naming_score_for_file_12
    naming_score_for_file_11 --> naming_score_for_file_13
    naming_score_for_file_12 --> naming_score_for_file_13
    naming_score_for_file_13 --> naming_score_for_file_14
    naming_score_for_file_14 --> naming_score_for_file_15
    naming_score_for_file_15 --> naming_score_for_file_16
    naming_score_for_file_14 --> naming_score_for_file_17
    naming_score_for_file_16 --> naming_score_for_file_18
    naming_score_for_file_17 --> naming_score_for_file_18
    naming_score_for_file_18 --> naming_score_for_file_19
    naming_score_for_file_19 --> naming_score_for_file_20
    naming_score_for_file_20 --> naming_score_for_file_21
    naming_score_for_file_19 --> naming_score_for_file_22
    naming_score_for_file_21 --> naming_score_for_file_23
    naming_score_for_file_22 --> naming_score_for_file_23
    naming_score_for_file_23 --> naming_score_for_file_24
    naming_score_for_file_24 --> naming_score_for_file_25
    naming_score_for_file_25 --> naming_score_for_file_26
    naming_score_for_file_24 --> naming_score_for_file_27
    naming_score_for_file_26 --> naming_score_for_file_28
    naming_score_for_file_27 --> naming_score_for_file_28
    naming_score_for_file_28 --> naming_score_for_file_29
    naming_score_for_file_29 --> naming_score_for_file_30
    naming_score_for_file_30 --> naming_score_for_file_31
    naming_score_for_file_31 --> naming_score_for_file_32
    naming_score_for_file_32 --> naming_score_for_file_33
    naming_score_for_file_33 --> naming_score_for_file_34
    naming_score_for_file_34 --> naming_score_for_file_35
    naming_score_for_file_33 --> naming_score_for_file_36
    naming_score_for_file_36 --> naming_score_for_file_37
    naming_score_for_file_35 --> naming_score_for_file_38
    naming_score_for_file_37 --> naming_score_for_file_38
    naming_score_for_file_29 --> naming_score_for_file_39
    naming_score_for_file_38 --> naming_score_for_file_40
    naming_score_for_file_39 --> naming_score_for_file_40
    naming_score_for_file_40 --> naming_score_for_file_41
    naming_score_for_file_41 --> naming_score_for_file_42
    naming_score_for_file_42 --> naming_score_for_file_43
    naming_score_for_file_43 --> naming_score_for_file_44
    naming_score_for_file_44 --> naming_score_for_file_45
    naming_score_for_file_45 --> naming_score_for_file_46
    naming_score_for_file_46 --> naming_score_for_file_47
    naming_score_for_file_47 --> naming_score_for_file_48
    naming_score_for_file_48 --> naming_score_for_file_49
    naming_score_for_file_49 --> naming_score_for_file_50
    naming_score_for_file_50 --> naming_score_for_file_51
    naming_score_for_file_51 --> naming_score_for_file_52
    naming_score_for_file_52 --> naming_score_for_file_53
    naming_score_for_file_51 --> naming_score_for_file_54
    naming_score_for_file_54 --> naming_score_for_file_55
    naming_score_for_file_55 --> naming_score_for_file_56
    naming_score_for_file_56 --> naming_score_for_file_57
    naming_score_for_file_55 --> naming_score_for_file_58
    naming_score_for_file_57 --> naming_score_for_file_59
    naming_score_for_file_58 --> naming_score_for_file_59
    naming_score_for_file_53 --> naming_score_for_file_60
    naming_score_for_file_59 --> naming_score_for_file_60
    naming_score_for_file_41 --> naming_score_for_file_61
    naming_score_for_file_60 --> naming_score_for_file_62
    naming_score_for_file_61 --> naming_score_for_file_62
    naming_score_for_file_62 --> naming_score_for_file_63
    naming_score_for_file_63 --> naming_score_for_file_64
    naming_score_for_file_64 --> naming_score_for_file_65
    naming_score_for_file_63 --> naming_score_for_file_66
    naming_score_for_file_65 --> naming_score_for_file_67
    naming_score_for_file_66 --> naming_score_for_file_67
    naming_score_for_file_67 --> naming_score_for_file_68
    naming_score_for_file_68 --> naming_score_for_file_69
    naming_score_for_file_69 --> naming_score_for_file_70
    naming_score_for_file_68 --> naming_score_for_file_71
    naming_score_for_file_70 --> naming_score_for_file_72
    naming_score_for_file_71 --> naming_score_for_file_72
    naming_score_for_file_72 --> naming_score_for_file_73
    naming_score_for_file_73 --> naming_score_for_file_74
```

## Function: `order_julia_files_by_dependency`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    order_julia_files_by_dependency_0["ENTRY"]
    order_julia_files_by_dependency_1["use"]
    order_julia_files_by_dependency_2["use"]
    order_julia_files_by_dependency_3["let mut file_layers : HashMap < PathBuf , String > = HashMap :: new ()"]
    order_julia_files_by_dependency_4["let mut nodes : BTreeSet < String > = BTreeSet :: new ()"]
    order_julia_files_by_dependency_5["let mut edges_map : BTreeMap < (String , String) , BTreeSet < ReferenceDetail > > = BTreeMap :: new ()"]
    order_julia_files_by_dependency_6["let mut unresolved = Vec :: new ()"]
    order_julia_files_by_dependency_7["let resolver = LayerResolver :: build (root) ?"]
    order_julia_files_by_dependency_8["let entry_files = crate :: cluster_001 :: julia_entry_paths (root)"]
    order_julia_files_by_dependency_9["for file in files { let layer = crate :: cluster_001 :: detect_layer (file) ;..."]
    order_julia_files_by_dependency_10["crate :: cluster_008 :: build_result (files , file_layers , nodes , edges_map..."]
    order_julia_files_by_dependency_11["EXIT"]
    order_julia_files_by_dependency_0 --> order_julia_files_by_dependency_1
    order_julia_files_by_dependency_1 --> order_julia_files_by_dependency_2
    order_julia_files_by_dependency_2 --> order_julia_files_by_dependency_3
    order_julia_files_by_dependency_3 --> order_julia_files_by_dependency_4
    order_julia_files_by_dependency_4 --> order_julia_files_by_dependency_5
    order_julia_files_by_dependency_5 --> order_julia_files_by_dependency_6
    order_julia_files_by_dependency_6 --> order_julia_files_by_dependency_7
    order_julia_files_by_dependency_7 --> order_julia_files_by_dependency_8
    order_julia_files_by_dependency_8 --> order_julia_files_by_dependency_9
    order_julia_files_by_dependency_9 --> order_julia_files_by_dependency_10
    order_julia_files_by_dependency_10 --> order_julia_files_by_dependency_11
```

## Function: `order_rust_files_by_dependency`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    order_rust_files_by_dependency_0["ENTRY"]
    order_rust_files_by_dependency_1["let module_map = crate :: cluster_010 :: build_module_root_map (root) ?"]
    order_rust_files_by_dependency_2["let entry_files = rust_entry_paths (root)"]
    order_rust_files_by_dependency_3["let mut file_layers : HashMap < PathBuf , String > = HashMap :: new ()"]
    order_rust_files_by_dependency_4["let mut nodes : BTreeSet < String > = BTreeSet :: new ()"]
    order_rust_files_by_dependency_5["let mut edges_map : BTreeMap < (String , String) , BTreeSet < ReferenceDetail > > = BTreeMap :: new ()"]
    order_rust_files_by_dependency_6["let mut unresolved = Vec :: new ()"]
    order_rust_files_by_dependency_7["for file in files { let layer = detect_layer (file) ; nodes . insert (layer ...."]
    order_rust_files_by_dependency_8["crate :: cluster_008 :: build_result (files , file_layers , nodes , edges_map..."]
    order_rust_files_by_dependency_9["EXIT"]
    order_rust_files_by_dependency_0 --> order_rust_files_by_dependency_1
    order_rust_files_by_dependency_1 --> order_rust_files_by_dependency_2
    order_rust_files_by_dependency_2 --> order_rust_files_by_dependency_3
    order_rust_files_by_dependency_3 --> order_rust_files_by_dependency_4
    order_rust_files_by_dependency_4 --> order_rust_files_by_dependency_5
    order_rust_files_by_dependency_5 --> order_rust_files_by_dependency_6
    order_rust_files_by_dependency_6 --> order_rust_files_by_dependency_7
    order_rust_files_by_dependency_7 --> order_rust_files_by_dependency_8
    order_rust_files_by_dependency_8 --> order_rust_files_by_dependency_9
```

## Function: `ordered_by_name`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    ordered_by_name_0["ENTRY"]
    ordered_by_name_1["let mut sorted = files . to_vec ()"]
    ordered_by_name_2["sorted . sort ()"]
    ordered_by_name_3["sorted . into_iter () . filter_map (| path | node_map . get (& path) . copied..."]
    ordered_by_name_4["EXIT"]
    ordered_by_name_0 --> ordered_by_name_1
    ordered_by_name_1 --> ordered_by_name_2
    ordered_by_name_2 --> ordered_by_name_3
    ordered_by_name_3 --> ordered_by_name_4
```

## Function: `run_analysis`

- File: src/000_dependency_analysis.rs
- Branches: 4
- Loops: 0
- Nodes: 88
- Edges: 91

```mermaid
flowchart TD
    run_analysis_0["ENTRY"]
    run_analysis_1["use"]
    run_analysis_2["use"]
    run_analysis_3["use"]
    run_analysis_4["use"]
    run_analysis_5["use"]
    run_analysis_6["use"]
    run_analysis_7["use"]
    run_analysis_8["use"]
    run_analysis_9["use"]
    run_analysis_10["let julia_script_path = root_path . join ('src/000_main.jl')"]
    run_analysis_11["macro println"]
    run_analysis_12["macro println"]
    run_analysis_13["macro println"]
    run_analysis_14["macro println"]
    run_analysis_15["macro println"]
    run_analysis_16["let rust_analyzer = RustAnalyzer :: new (root_path . to_string_lossy () . to_string ())"]
    run_analysis_17["let mut combined_result = AnalysisResult :: new ()"]
    run_analysis_18["macro println"]
    run_analysis_19["let mut rust_count = 0"]
    run_analysis_20["let rust_files = gather_rust_files (root_path)"]
    run_analysis_21["let (ordered_rust_files , rust_layer_graph) = crate :: dependency :: order_rust_files_by_dependency (& rust_files , root_pa..."]
    run_analysis_22["let rust_file_ordering = crate :: dependency :: analyze_file_ordering (& rust_files , None) . context ..."]
    run_analysis_23["let julia_file_ordering = FileOrderingResult { ordered_files : Vec :: new () , violations : Vec :: new ..."]
    run_analysis_24["for path in ordered_rust_files { if verbose { println ! (' Analyzing: {:?}' ,..."]
    run_analysis_25["macro println"]
    run_analysis_26["let mut julia_count = 0"]
    run_analysis_27["let mut julia_layer_graph = LayerGraph { ordered_layers : Vec :: new () , edges : Vec :: new () , cycles ..."]
    run_analysis_28["if ! skip_julia"]
    run_analysis_29["THEN BB"]
    run_analysis_30["macro println"]
    run_analysis_31["let julia_files = gather_julia_files (root_path)"]
    run_analysis_32["let (ordered_julia_files , jlg) = crate :: dependency :: order_julia_files_by_dependency (& julia_files , root_..."]
    run_analysis_33["julia_layer_graph = jlg"]
    run_analysis_34["if julia_script_path . exists ()"]
    run_analysis_35["THEN BB"]
    run_analysis_36["let julia_analyzer = JuliaAnalyzer :: new (root_path . to_path_buf () , julia_script_path . clone ..."]
    run_analysis_37["for path in ordered_julia_files { if verbose { println ! (' Analyzing: {:?}' ..."]
    run_analysis_38["ELSE BB"]
    run_analysis_39["{ println ! (' Skipping Julia analysis (script not found)') ; }"]
    run_analysis_40["IF JOIN"]
    run_analysis_41["macro println"]
    run_analysis_42["EMPTY ELSE"]
    run_analysis_43["IF JOIN"]
    run_analysis_44["if dead_code || dead_code_filter || dead_code_json . is_some () || dead_code_..."]
    run_analysis_45["THEN BB"]
    run_analysis_46["let policy = if let Some (policy_path) = dead_code_policy { Some (crate :: dead_code_polic..."]
    run_analysis_47["let config = crate :: dead_code_cli :: DeadCodeRunConfig { root : root_path . to_path_buf ..."]
    run_analysis_48["let report = crate :: dead_code_cli :: run_dead_code_pipeline (& combined_result . element..."]
    run_analysis_49["if dead_code_filter"]
    run_analysis_50["THEN BB"]
    run_analysis_51["combined_result . elements = crate :: dead_code_filter :: filter_dead_code_el..."]
    run_analysis_52["EMPTY ELSE"]
    run_analysis_53["IF JOIN"]
    run_analysis_54["EMPTY ELSE"]
    run_analysis_55["IF JOIN"]
    run_analysis_56["macro println"]
    run_analysis_57["let mut cf_analyzer = ControlFlowAnalyzer :: new ()"]
    run_analysis_58["cf_analyzer . build_call_graph (& combined_result)"]
    run_analysis_59["use"]
    run_analysis_60["macro println"]
    run_analysis_61["let invariants_result = { let invariant_detector = InvariantDetector :: new (& combined_result , & co..."]
    run_analysis_62["let constraints = { let invariant_detector = InvariantDetector :: new (& combined_result , & co..."]
    run_analysis_63["combined_result . invariants = invariants_result"]
    run_analysis_64["combined_result . constraints = constraints"]
    run_analysis_65["macro println"]
    run_analysis_66["let cohesion_analyzer = FunctionCohesionAnalyzer :: new ()"]
    run_analysis_67["let placements = cohesion_analyzer . analyze (& combined_result) ?"]
    run_analysis_68["let clusters = cohesion_analyzer . detect_clusters (& combined_result) ?"]
    run_analysis_69["macro println"]
    run_analysis_70["let dir_analyzer = DirectoryAnalyzer :: new (root_path . to_path_buf ())"]
    run_analysis_71["let dir_analysis = dir_analyzer . analyze () ?"]
    run_analysis_72["macro println"]
    run_analysis_73["let report_gen = ReportGenerator :: new (output_path . to_string_lossy () . to_string ())"]
    run_analysis_74["report_gen . generate_all (& combined_result , & cf_analyzer , & rust_layer_g..."]
    run_analysis_75["macro println"]
    run_analysis_76["export_program_cfg_to_path (& combined_result , & cf_analyzer . call_edges ()..."]
    run_analysis_77["macro println"]
    run_analysis_78["use"]
    run_analysis_79["invariant_reporter :: generate_invariant_report (& combined_result . invarian..."]
    run_analysis_80["invariant_reporter :: export_constraints_json (& combined_result . constraint..."]
    run_analysis_81["macro println"]
    run_analysis_82["macro println"]
    run_analysis_83["macro println"]
    run_analysis_84["macro println"]
    run_analysis_85["macro println"]
    run_analysis_86["Ok (())"]
    run_analysis_87["EXIT"]
    run_analysis_0 --> run_analysis_1
    run_analysis_1 --> run_analysis_2
    run_analysis_2 --> run_analysis_3
    run_analysis_3 --> run_analysis_4
    run_analysis_4 --> run_analysis_5
    run_analysis_5 --> run_analysis_6
    run_analysis_6 --> run_analysis_7
    run_analysis_7 --> run_analysis_8
    run_analysis_8 --> run_analysis_9
    run_analysis_9 --> run_analysis_10
    run_analysis_10 --> run_analysis_11
    run_analysis_11 --> run_analysis_12
    run_analysis_12 --> run_analysis_13
    run_analysis_13 --> run_analysis_14
    run_analysis_14 --> run_analysis_15
    run_analysis_15 --> run_analysis_16
    run_analysis_16 --> run_analysis_17
    run_analysis_17 --> run_analysis_18
    run_analysis_18 --> run_analysis_19
    run_analysis_19 --> run_analysis_20
    run_analysis_20 --> run_analysis_21
    run_analysis_21 --> run_analysis_22
    run_analysis_22 --> run_analysis_23
    run_analysis_23 --> run_analysis_24
    run_analysis_24 --> run_analysis_25
    run_analysis_25 --> run_analysis_26
    run_analysis_26 --> run_analysis_27
    run_analysis_27 --> run_analysis_28
    run_analysis_28 --> run_analysis_29
    run_analysis_29 --> run_analysis_30
    run_analysis_30 --> run_analysis_31
    run_analysis_31 --> run_analysis_32
    run_analysis_32 --> run_analysis_33
    run_analysis_33 --> run_analysis_34
    run_analysis_34 --> run_analysis_35
    run_analysis_35 --> run_analysis_36
    run_analysis_36 --> run_analysis_37
    run_analysis_34 --> run_analysis_38
    run_analysis_38 --> run_analysis_39
    run_analysis_37 --> run_analysis_40
    run_analysis_39 --> run_analysis_40
    run_analysis_40 --> run_analysis_41
    run_analysis_28 --> run_analysis_42
    run_analysis_41 --> run_analysis_43
    run_analysis_42 --> run_analysis_43
    run_analysis_43 --> run_analysis_44
    run_analysis_44 --> run_analysis_45
    run_analysis_45 --> run_analysis_46
    run_analysis_46 --> run_analysis_47
    run_analysis_47 --> run_analysis_48
    run_analysis_48 --> run_analysis_49
    run_analysis_49 --> run_analysis_50
    run_analysis_50 --> run_analysis_51
    run_analysis_49 --> run_analysis_52
    run_analysis_51 --> run_analysis_53
    run_analysis_52 --> run_analysis_53
    run_analysis_44 --> run_analysis_54
    run_analysis_53 --> run_analysis_55
    run_analysis_54 --> run_analysis_55
    run_analysis_55 --> run_analysis_56
    run_analysis_56 --> run_analysis_57
    run_analysis_57 --> run_analysis_58
    run_analysis_58 --> run_analysis_59
    run_analysis_59 --> run_analysis_60
    run_analysis_60 --> run_analysis_61
    run_analysis_61 --> run_analysis_62
    run_analysis_62 --> run_analysis_63
    run_analysis_63 --> run_analysis_64
    run_analysis_64 --> run_analysis_65
    run_analysis_65 --> run_analysis_66
    run_analysis_66 --> run_analysis_67
    run_analysis_67 --> run_analysis_68
    run_analysis_68 --> run_analysis_69
    run_analysis_69 --> run_analysis_70
    run_analysis_70 --> run_analysis_71
    run_analysis_71 --> run_analysis_72
    run_analysis_72 --> run_analysis_73
    run_analysis_73 --> run_analysis_74
    run_analysis_74 --> run_analysis_75
    run_analysis_75 --> run_analysis_76
    run_analysis_76 --> run_analysis_77
    run_analysis_77 --> run_analysis_78
    run_analysis_78 --> run_analysis_79
    run_analysis_79 --> run_analysis_80
    run_analysis_80 --> run_analysis_81
    run_analysis_81 --> run_analysis_82
    run_analysis_82 --> run_analysis_83
    run_analysis_83 --> run_analysis_84
    run_analysis_84 --> run_analysis_85
    run_analysis_85 --> run_analysis_86
    run_analysis_86 --> run_analysis_87
```

## Function: `rust_entry_paths`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    rust_entry_paths_0["ENTRY"]
    rust_entry_paths_1["let src_dir = crate :: layer_utilities :: resolve_source_root (root)"]
    rust_entry_paths_2["['lib.rs' , 'main.rs'] . iter () . map (| rel | src_dir . join (rel)) . filte..."]
    rust_entry_paths_3["EXIT"]
    rust_entry_paths_0 --> rust_entry_paths_1
    rust_entry_paths_1 --> rust_entry_paths_2
    rust_entry_paths_2 --> rust_entry_paths_3
```

## Function: `temp_dir`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    temp_dir_0["ENTRY"]
    temp_dir_1["let mut dir = std :: env :: temp_dir ()"]
    temp_dir_2["dir . push (format ! ('mmsb_analyzer_{}_{}' , name , std :: time :: SystemTim..."]
    temp_dir_3["dir"]
    temp_dir_4["EXIT"]
    temp_dir_0 --> temp_dir_1
    temp_dir_1 --> temp_dir_2
    temp_dir_2 --> temp_dir_3
    temp_dir_3 --> temp_dir_4
```

## Function: `test_detects_cycles`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    test_detects_cycles_0["ENTRY"]
    test_detects_cycles_1["detects_cycles () . unwrap ()"]
    test_detects_cycles_2["EXIT"]
    test_detects_cycles_0 --> test_detects_cycles_1
    test_detects_cycles_1 --> test_detects_cycles_2
```

## Function: `test_generates_canonical_names_and_violations`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    test_generates_canonical_names_and_violations_0["ENTRY"]
    test_generates_canonical_names_and_violations_1["generates_canonical_names_and_violations () . unwrap ()"]
    test_generates_canonical_names_and_violations_2["EXIT"]
    test_generates_canonical_names_and_violations_0 --> test_generates_canonical_names_and_violations_1
    test_generates_canonical_names_and_violations_1 --> test_generates_canonical_names_and_violations_2
```

## Function: `topo_sort_orders_dependencies`

- File: src/000_dependency_analysis.rs
- Branches: 0
- Loops: 0
- Nodes: 17
- Edges: 16

```mermaid
flowchart TD
    topo_sort_orders_dependencies_0["ENTRY"]
    topo_sort_orders_dependencies_1["use"]
    topo_sort_orders_dependencies_2["use"]
    topo_sort_orders_dependencies_3["let dir = temp_dir ('topo')"]
    topo_sort_orders_dependencies_4["create_dir_all (& dir) ?"]
    topo_sort_orders_dependencies_5["let a = dir . join ('a.rs')"]
    topo_sort_orders_dependencies_6["let b = dir . join ('b.rs')"]
    topo_sort_orders_dependencies_7["let c = dir . join ('c.rs')"]
    topo_sort_orders_dependencies_8["write (& a , 'pub fn a() {}') ?"]
    topo_sort_orders_dependencies_9["write (& b , 'use crate::a; pub fn b() {}') ?"]
    topo_sort_orders_dependencies_10["write (& c , 'use crate::b; pub fn c() {}') ?"]
    topo_sort_orders_dependencies_11["let result = analyze_file_ordering (& [c . clone () , b . clone () , a . clone ()] , Some ..."]
    topo_sort_orders_dependencies_12["let ordered : Vec < _ > = result . ordered_files . iter () . map (| entry | entry . current_path . clon..."]
    topo_sort_orders_dependencies_13["macro assert_eq"]
    topo_sort_orders_dependencies_14["macro assert"]
    topo_sort_orders_dependencies_15["Ok (())"]
    topo_sort_orders_dependencies_16["EXIT"]
    topo_sort_orders_dependencies_0 --> topo_sort_orders_dependencies_1
    topo_sort_orders_dependencies_1 --> topo_sort_orders_dependencies_2
    topo_sort_orders_dependencies_2 --> topo_sort_orders_dependencies_3
    topo_sort_orders_dependencies_3 --> topo_sort_orders_dependencies_4
    topo_sort_orders_dependencies_4 --> topo_sort_orders_dependencies_5
    topo_sort_orders_dependencies_5 --> topo_sort_orders_dependencies_6
    topo_sort_orders_dependencies_6 --> topo_sort_orders_dependencies_7
    topo_sort_orders_dependencies_7 --> topo_sort_orders_dependencies_8
    topo_sort_orders_dependencies_8 --> topo_sort_orders_dependencies_9
    topo_sort_orders_dependencies_9 --> topo_sort_orders_dependencies_10
    topo_sort_orders_dependencies_10 --> topo_sort_orders_dependencies_11
    topo_sort_orders_dependencies_11 --> topo_sort_orders_dependencies_12
    topo_sort_orders_dependencies_12 --> topo_sort_orders_dependencies_13
    topo_sort_orders_dependencies_13 --> topo_sort_orders_dependencies_14
    topo_sort_orders_dependencies_14 --> topo_sort_orders_dependencies_15
    topo_sort_orders_dependencies_15 --> topo_sort_orders_dependencies_16
```

## Function: `topo_sort_within`

- File: src/000_dependency_analysis.rs
- Branches: 1
- Loops: 0
- Nodes: 16
- Edges: 16

```mermaid
flowchart TD
    topo_sort_within_0["ENTRY"]
    topo_sort_within_1["let node_set : HashSet < NodeIndex > = nodes . iter () . copied () . collect ()"]
    topo_sort_within_2["let mut indegree : HashMap < NodeIndex , usize > = HashMap :: new ()"]
    topo_sort_within_3["for & node in nodes { indegree . insert (node , 0) ; }"]
    topo_sort_within_4["for & node in nodes { let incoming = graph . neighbors_directed (node , petgr..."]
    topo_sort_within_5["let mut queue = std :: collections :: VecDeque :: new ()"]
    topo_sort_within_6["for & node in nodes { if indegree . get (& node) . copied () . unwrap_or (0) ..."]
    topo_sort_within_7["let mut ordered = Vec :: new ()"]
    topo_sort_within_8["while let Some (node) = queue . pop_front () { ordered . push (node) ; for ne..."]
    topo_sort_within_9["if ordered . len () != nodes . len ()"]
    topo_sort_within_10["THEN BB"]
    topo_sort_within_11["return Err (anyhow :: anyhow ! ('Cycle detected within layer group'))"]
    topo_sort_within_12["EMPTY ELSE"]
    topo_sort_within_13["IF JOIN"]
    topo_sort_within_14["Ok (ordered)"]
    topo_sort_within_15["EXIT"]
    topo_sort_within_0 --> topo_sort_within_1
    topo_sort_within_1 --> topo_sort_within_2
    topo_sort_within_2 --> topo_sort_within_3
    topo_sort_within_3 --> topo_sort_within_4
    topo_sort_within_4 --> topo_sort_within_5
    topo_sort_within_5 --> topo_sort_within_6
    topo_sort_within_6 --> topo_sort_within_7
    topo_sort_within_7 --> topo_sort_within_8
    topo_sort_within_8 --> topo_sort_within_9
    topo_sort_within_9 --> topo_sort_within_10
    topo_sort_within_10 --> topo_sort_within_11
    topo_sort_within_9 --> topo_sort_within_12
    topo_sort_within_11 --> topo_sort_within_13
    topo_sort_within_12 --> topo_sort_within_13
    topo_sort_within_13 --> topo_sort_within_14
    topo_sort_within_14 --> topo_sort_within_15
```

## Function: `topological_sort`

- File: src/000_dependency_analysis.rs
- Branches: 1
- Loops: 0
- Nodes: 16
- Edges: 16

```mermaid
flowchart TD
    topological_sort_0["ENTRY"]
    topological_sort_1["use"]
    topological_sort_2["use"]
    topological_sort_3["let mut indegree = vec ! [0usize ; graph . node_count ()]"]
    topological_sort_4["for node in graph . node_indices () { indegree [node . index ()] = graph . ne..."]
    topological_sort_5["let mut queue = VecDeque :: new ()"]
    topological_sort_6["for node in graph . node_indices () { if indegree [node . index ()] == 0 { qu..."]
    topological_sort_7["let mut ordered = Vec :: new ()"]
    topological_sort_8["while let Some (node) = queue . pop_front () { ordered . push (node) ; for ne..."]
    topological_sort_9["if ordered . len () != graph . node_count ()"]
    topological_sort_10["THEN BB"]
    topological_sort_11["return Err (anyhow :: anyhow ! ('Cycle detected in dependency graph'))"]
    topological_sort_12["EMPTY ELSE"]
    topological_sort_13["IF JOIN"]
    topological_sort_14["Ok (ordered)"]
    topological_sort_15["EXIT"]
    topological_sort_0 --> topological_sort_1
    topological_sort_1 --> topological_sort_2
    topological_sort_2 --> topological_sort_3
    topological_sort_3 --> topological_sort_4
    topological_sort_4 --> topological_sort_5
    topological_sort_5 --> topological_sort_6
    topological_sort_6 --> topological_sort_7
    topological_sort_7 --> topological_sort_8
    topological_sort_8 --> topological_sort_9
    topological_sort_9 --> topological_sort_10
    topological_sort_10 --> topological_sort_11
    topological_sort_9 --> topological_sort_12
    topological_sort_11 --> topological_sort_13
    topological_sort_12 --> topological_sort_13
    topological_sort_13 --> topological_sort_14
    topological_sort_14 --> topological_sort_15
```

