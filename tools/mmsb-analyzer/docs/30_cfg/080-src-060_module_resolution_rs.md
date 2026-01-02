# CFG Group: src/060_module_resolution.rs

## Function: `build_dependency_map`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    build_dependency_map_0["ENTRY"]
    build_dependency_map_1["let mut dep_map : HashMap < PathBuf , Vec < PathBuf > > = HashMap :: new ()"]
    build_dependency_map_2["for file in files { let deps = extract_dependencies (file , file_set , module..."]
    build_dependency_map_3["Ok (dep_map)"]
    build_dependency_map_4["EXIT"]
    build_dependency_map_0 --> build_dependency_map_1
    build_dependency_map_1 --> build_dependency_map_2
    build_dependency_map_2 --> build_dependency_map_3
    build_dependency_map_3 --> build_dependency_map_4
```

## Function: `build_module_root_map`

- File: src/060_module_resolution.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    build_module_root_map_0["ENTRY"]
    build_module_root_map_1["let src_dir = root . join ('src')"]
    build_module_root_map_2["let mut map = HashMap :: new ()"]
    build_module_root_map_3["if src_dir . is_dir ()"]
    build_module_root_map_4["THEN BB"]
    build_module_root_map_5["for entry in fs :: read_dir (& src_dir) ? { let entry = entry ? ; let path = ..."]
    build_module_root_map_6["EMPTY ELSE"]
    build_module_root_map_7["IF JOIN"]
    build_module_root_map_8["Ok (map)"]
    build_module_root_map_9["EXIT"]
    build_module_root_map_0 --> build_module_root_map_1
    build_module_root_map_1 --> build_module_root_map_2
    build_module_root_map_2 --> build_module_root_map_3
    build_module_root_map_3 --> build_module_root_map_4
    build_module_root_map_4 --> build_module_root_map_5
    build_module_root_map_3 --> build_module_root_map_6
    build_module_root_map_5 --> build_module_root_map_7
    build_module_root_map_6 --> build_module_root_map_7
    build_module_root_map_7 --> build_module_root_map_8
    build_module_root_map_8 --> build_module_root_map_9
```

## Function: `contains_tools`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    contains_tools_0["ENTRY"]
    contains_tools_1["path . components () . any (| c | c . as_os_str () == 'tools')"]
    contains_tools_2["EXIT"]
    contains_tools_0 --> contains_tools_1
    contains_tools_1 --> contains_tools_2
```

## Function: `extract_dependencies`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    extract_dependencies_0["ENTRY"]
    extract_dependencies_1["let ext = file . extension () . and_then (| s | s . to_str ()) . unwrap_or ('')"]
    extract_dependencies_2["match ext { 'rs' => extract_rust_dependencies (file , file_set , module_map) ..."]
    extract_dependencies_3["EXIT"]
    extract_dependencies_0 --> extract_dependencies_1
    extract_dependencies_1 --> extract_dependencies_2
    extract_dependencies_2 --> extract_dependencies_3
```

## Function: `extract_julia_dependencies`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    extract_julia_dependencies_0["ENTRY"]
    extract_julia_dependencies_1["item"]
    extract_julia_dependencies_2["item"]
    extract_julia_dependencies_3["item"]
    extract_julia_dependencies_4["item"]
    extract_julia_dependencies_5["item"]
    extract_julia_dependencies_6["item"]
    extract_julia_dependencies_7["let content = fs :: read_to_string (file) . with_context (| | format ! ('Unable to read {:?..."]
    extract_julia_dependencies_8["let mut deps = Vec :: new ()"]
    extract_julia_dependencies_9["for cap in INCLUDE_RE . captures_iter (& content) { if let Some (path_match) ..."]
    extract_julia_dependencies_10["for cap in MMSB_USING_RE . captures_iter (& content) { if let Some (module_ma..."]
    extract_julia_dependencies_11["for cap in MMSB_SYMBOL_RE . captures_iter (& content) { if let Some (symbols)..."]
    extract_julia_dependencies_12["for cap in LOCAL_USING_RE . captures_iter (& content) { if let Some (module_m..."]
    extract_julia_dependencies_13["for cap in PLAIN_USING_RE . captures_iter (& content) { if let Some (module_m..."]
    extract_julia_dependencies_14["Ok (deps)"]
    extract_julia_dependencies_15["EXIT"]
    extract_julia_dependencies_0 --> extract_julia_dependencies_1
    extract_julia_dependencies_1 --> extract_julia_dependencies_2
    extract_julia_dependencies_2 --> extract_julia_dependencies_3
    extract_julia_dependencies_3 --> extract_julia_dependencies_4
    extract_julia_dependencies_4 --> extract_julia_dependencies_5
    extract_julia_dependencies_5 --> extract_julia_dependencies_6
    extract_julia_dependencies_6 --> extract_julia_dependencies_7
    extract_julia_dependencies_7 --> extract_julia_dependencies_8
    extract_julia_dependencies_8 --> extract_julia_dependencies_9
    extract_julia_dependencies_9 --> extract_julia_dependencies_10
    extract_julia_dependencies_10 --> extract_julia_dependencies_11
    extract_julia_dependencies_11 --> extract_julia_dependencies_12
    extract_julia_dependencies_12 --> extract_julia_dependencies_13
    extract_julia_dependencies_13 --> extract_julia_dependencies_14
    extract_julia_dependencies_14 --> extract_julia_dependencies_15
```

## Function: `extract_rust_dependencies`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    extract_rust_dependencies_0["ENTRY"]
    extract_rust_dependencies_1["struct UseCollector"]
    extract_rust_dependencies_2["impl block"]
    extract_rust_dependencies_3["let content = fs :: read_to_string (file) . with_context (| | format ! ('Unable to read {:?..."]
    extract_rust_dependencies_4["let syntax = syn :: parse_file (& content) . with_context (| | format ! ('Unable to parse ..."]
    extract_rust_dependencies_5["let mut collector = UseCollector :: default ()"]
    extract_rust_dependencies_6["collector . visit_file (& syntax)"]
    extract_rust_dependencies_7["let mut deps = Vec :: new ()"]
    extract_rust_dependencies_8["for root in collector . roots { if let Some (path) = resolve_module (& root ,..."]
    extract_rust_dependencies_9["for module in collector . mods { if let Some (path) = resolve_module (& modul..."]
    extract_rust_dependencies_10["Ok (deps)"]
    extract_rust_dependencies_11["EXIT"]
    extract_rust_dependencies_0 --> extract_rust_dependencies_1
    extract_rust_dependencies_1 --> extract_rust_dependencies_2
    extract_rust_dependencies_2 --> extract_rust_dependencies_3
    extract_rust_dependencies_3 --> extract_rust_dependencies_4
    extract_rust_dependencies_4 --> extract_rust_dependencies_5
    extract_rust_dependencies_5 --> extract_rust_dependencies_6
    extract_rust_dependencies_6 --> extract_rust_dependencies_7
    extract_rust_dependencies_7 --> extract_rust_dependencies_8
    extract_rust_dependencies_8 --> extract_rust_dependencies_9
    extract_rust_dependencies_9 --> extract_rust_dependencies_10
    extract_rust_dependencies_10 --> extract_rust_dependencies_11
```

## Function: `normalize_module_name`

- File: src/060_module_resolution.rs
- Branches: 2
- Loops: 0
- Nodes: 12
- Edges: 13

```mermaid
flowchart TD
    normalize_module_name_0["ENTRY"]
    normalize_module_name_1["if let Some (pos) = name . find ('_')"]
    normalize_module_name_2["THEN BB"]
    normalize_module_name_3["if name [.. pos] . chars () . all (| c | c . is_ascii_digit ())"]
    normalize_module_name_4["THEN BB"]
    normalize_module_name_5["return name [pos + 1 ..] . to_string ()"]
    normalize_module_name_6["EMPTY ELSE"]
    normalize_module_name_7["IF JOIN"]
    normalize_module_name_8["EMPTY ELSE"]
    normalize_module_name_9["IF JOIN"]
    normalize_module_name_10["name . to_string ()"]
    normalize_module_name_11["EXIT"]
    normalize_module_name_0 --> normalize_module_name_1
    normalize_module_name_1 --> normalize_module_name_2
    normalize_module_name_2 --> normalize_module_name_3
    normalize_module_name_3 --> normalize_module_name_4
    normalize_module_name_4 --> normalize_module_name_5
    normalize_module_name_3 --> normalize_module_name_6
    normalize_module_name_5 --> normalize_module_name_7
    normalize_module_name_6 --> normalize_module_name_7
    normalize_module_name_1 --> normalize_module_name_8
    normalize_module_name_7 --> normalize_module_name_9
    normalize_module_name_8 --> normalize_module_name_9
    normalize_module_name_9 --> normalize_module_name_10
    normalize_module_name_10 --> normalize_module_name_11
```

## Function: `resolve_module`

- File: src/060_module_resolution.rs
- Branches: 1
- Loops: 0
- Nodes: 9
- Edges: 9

```mermaid
flowchart TD
    resolve_module_0["ENTRY"]
    resolve_module_1["let key = normalize_module_name (root)"]
    resolve_module_2["if let Some (path) = module_map . get (& key)"]
    resolve_module_3["THEN BB"]
    resolve_module_4["return Some (path . clone ())"]
    resolve_module_5["EMPTY ELSE"]
    resolve_module_6["IF JOIN"]
    resolve_module_7["module_map . iter () . find (| (name , _) | name == & & key) . map (| (_ , pa..."]
    resolve_module_8["EXIT"]
    resolve_module_0 --> resolve_module_1
    resolve_module_1 --> resolve_module_2
    resolve_module_2 --> resolve_module_3
    resolve_module_3 --> resolve_module_4
    resolve_module_2 --> resolve_module_5
    resolve_module_4 --> resolve_module_6
    resolve_module_5 --> resolve_module_6
    resolve_module_6 --> resolve_module_7
    resolve_module_7 --> resolve_module_8
```

## Function: `resolve_module_name`

- File: src/060_module_resolution.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    resolve_module_name_0["ENTRY"]
    resolve_module_name_1["let primary = module . split ('.') . next () . unwrap_or (module)"]
    resolve_module_name_2["resolve_module (primary , file_set , module_map)"]
    resolve_module_name_3["EXIT"]
    resolve_module_name_0 --> resolve_module_name_1
    resolve_module_name_1 --> resolve_module_name_2
    resolve_module_name_2 --> resolve_module_name_3
```

