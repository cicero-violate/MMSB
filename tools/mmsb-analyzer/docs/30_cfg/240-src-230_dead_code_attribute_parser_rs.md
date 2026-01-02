# CFG Group: src/230_dead_code_attribute_parser.rs

## Function: `collect_latent_attrs`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    collect_latent_attrs_0["ENTRY"]
    collect_latent_attrs_1["let mut markers = Vec :: new ()"]
    collect_latent_attrs_2["for attr in attrs { if ! attr . path () . is_ident ('mmsb_latent') { continue..."]
    collect_latent_attrs_3["markers"]
    collect_latent_attrs_4["EXIT"]
    collect_latent_attrs_0 --> collect_latent_attrs_1
    collect_latent_attrs_1 --> collect_latent_attrs_2
    collect_latent_attrs_2 --> collect_latent_attrs_3
    collect_latent_attrs_3 --> collect_latent_attrs_4
```

## Function: `detect_intent_signals`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    detect_intent_signals_0["ENTRY"]
    detect_intent_signals_1["let attrs = parse_mmsb_latent_attr (file)"]
    detect_intent_signals_2["let doc_map = scan_doc_comments (file)"]
    detect_intent_signals_3["let docs = merge_doc_intent (doc_map)"]
    detect_intent_signals_4["let dir_map = planned_directory_intent (file , policy)"]
    detect_intent_signals_5["merge_intent_sources (attrs , docs , dir_map)"]
    detect_intent_signals_6["EXIT"]
    detect_intent_signals_0 --> detect_intent_signals_1
    detect_intent_signals_1 --> detect_intent_signals_2
    detect_intent_signals_2 --> detect_intent_signals_3
    detect_intent_signals_3 --> detect_intent_signals_4
    detect_intent_signals_4 --> detect_intent_signals_5
    detect_intent_signals_5 --> detect_intent_signals_6
```

## Function: `detect_test_modules`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    detect_test_modules_0["ENTRY"]
    detect_test_modules_1["let contents = std :: fs :: read_to_string (file) . unwrap_or_default ()"]
    detect_test_modules_2["let parsed = match syn :: parse_file (& contents) { Ok (file) => file , Err (_) => return ..."]
    detect_test_modules_3["let mut modules = HashSet :: new ()"]
    detect_test_modules_4["for item in & parsed . items { if let Item :: Mod (item_mod) = item { if is_c..."]
    detect_test_modules_5["modules"]
    detect_test_modules_6["EXIT"]
    detect_test_modules_0 --> detect_test_modules_1
    detect_test_modules_1 --> detect_test_modules_2
    detect_test_modules_2 --> detect_test_modules_3
    detect_test_modules_3 --> detect_test_modules_4
    detect_test_modules_4 --> detect_test_modules_5
    detect_test_modules_5 --> detect_test_modules_6
```

## Function: `detect_test_symbols`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    detect_test_symbols_0["ENTRY"]
    detect_test_symbols_1["let contents = std :: fs :: read_to_string (file) . unwrap_or_default ()"]
    detect_test_symbols_2["let parsed = match syn :: parse_file (& contents) { Ok (file) => file , Err (_) => return ..."]
    detect_test_symbols_3["let mut symbols = HashSet :: new ()"]
    detect_test_symbols_4["for item in & parsed . items { if let Item :: Fn (item_fn) = item { if has_te..."]
    detect_test_symbols_5["symbols"]
    detect_test_symbols_6["EXIT"]
    detect_test_symbols_0 --> detect_test_symbols_1
    detect_test_symbols_1 --> detect_test_symbols_2
    detect_test_symbols_2 --> detect_test_symbols_3
    detect_test_symbols_3 --> detect_test_symbols_4
    detect_test_symbols_4 --> detect_test_symbols_5
    detect_test_symbols_5 --> detect_test_symbols_6
```

## Function: `extract_attribute_value`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    extract_attribute_value_0["ENTRY"]
    extract_attribute_value_1["let mut found = None"]
    extract_attribute_value_2["let _ = attr . parse_nested_meta (| meta | { if meta . path . is_ident (key) { let va..."]
    extract_attribute_value_3["found"]
    extract_attribute_value_4["EXIT"]
    extract_attribute_value_0 --> extract_attribute_value_1
    extract_attribute_value_1 --> extract_attribute_value_2
    extract_attribute_value_2 --> extract_attribute_value_3
    extract_attribute_value_3 --> extract_attribute_value_4
```

## Function: `marker_from_str`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    marker_from_str_0["ENTRY"]
    marker_from_str_1["match raw . to_ascii_lowercase () . as_str () { 'planned' => IntentMarker :: ..."]
    marker_from_str_2["EXIT"]
    marker_from_str_0 --> marker_from_str_1
    marker_from_str_1 --> marker_from_str_2
```

## Function: `parse_mmsb_latent_attr`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    parse_mmsb_latent_attr_0["ENTRY"]
    parse_mmsb_latent_attr_1["let contents = std :: fs :: read_to_string (path) . unwrap_or_default ()"]
    parse_mmsb_latent_attr_2["let file = match syn :: parse_file (& contents) { Ok (file) => file , Err (_) => return ..."]
    parse_mmsb_latent_attr_3["let mut map : HashMap < String , Vec < IntentMetadata > > = HashMap :: new ()"]
    parse_mmsb_latent_attr_4["for item in & file . items { let Some (name) = item_name (item) else { contin..."]
    parse_mmsb_latent_attr_5["map"]
    parse_mmsb_latent_attr_6["EXIT"]
    parse_mmsb_latent_attr_0 --> parse_mmsb_latent_attr_1
    parse_mmsb_latent_attr_1 --> parse_mmsb_latent_attr_2
    parse_mmsb_latent_attr_2 --> parse_mmsb_latent_attr_3
    parse_mmsb_latent_attr_3 --> parse_mmsb_latent_attr_4
    parse_mmsb_latent_attr_4 --> parse_mmsb_latent_attr_5
    parse_mmsb_latent_attr_5 --> parse_mmsb_latent_attr_6
```

## Function: `scan_file_attributes`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    scan_file_attributes_0["ENTRY"]
    scan_file_attributes_1["let contents = std :: fs :: read_to_string (path) . unwrap_or_default ()"]
    scan_file_attributes_2["let file = match syn :: parse_file (& contents) { Ok (file) => file , Err (_) => return ..."]
    scan_file_attributes_3["let mut tags = Vec :: new ()"]
    scan_file_attributes_4["for item in & file . items { let Some (symbol) = item_name (item) else { cont..."]
    scan_file_attributes_5["tags"]
    scan_file_attributes_6["EXIT"]
    scan_file_attributes_0 --> scan_file_attributes_1
    scan_file_attributes_1 --> scan_file_attributes_2
    scan_file_attributes_2 --> scan_file_attributes_3
    scan_file_attributes_3 --> scan_file_attributes_4
    scan_file_attributes_4 --> scan_file_attributes_5
    scan_file_attributes_5 --> scan_file_attributes_6
```

## Function: `scan_intent_tags`

- File: src/230_dead_code_attribute_parser.rs
- Branches: 1
- Loops: 0
- Nodes: 13
- Edges: 13

```mermaid
flowchart TD
    scan_intent_tags_0["ENTRY"]
    scan_intent_tags_1["let mut tags = Vec :: new ()"]
    scan_intent_tags_2["let attrs = parse_mmsb_latent_attr (file)"]
    scan_intent_tags_3["for (symbol , items) in attrs { for meta in items { tags . push (IntentTag { ..."]
    scan_intent_tags_4["let doc_map = scan_doc_comments (file)"]
    scan_intent_tags_5["for (symbol , markers) in doc_map { for marker in markers { tags . push (Inte..."]
    scan_intent_tags_6["if check_planned_directory (file , policy)"]
    scan_intent_tags_7["THEN BB"]
    scan_intent_tags_8["for symbol in collect_symbols (file) { tags . push (IntentTag { symbol , file..."]
    scan_intent_tags_9["EMPTY ELSE"]
    scan_intent_tags_10["IF JOIN"]
    scan_intent_tags_11["tags"]
    scan_intent_tags_12["EXIT"]
    scan_intent_tags_0 --> scan_intent_tags_1
    scan_intent_tags_1 --> scan_intent_tags_2
    scan_intent_tags_2 --> scan_intent_tags_3
    scan_intent_tags_3 --> scan_intent_tags_4
    scan_intent_tags_4 --> scan_intent_tags_5
    scan_intent_tags_5 --> scan_intent_tags_6
    scan_intent_tags_6 --> scan_intent_tags_7
    scan_intent_tags_7 --> scan_intent_tags_8
    scan_intent_tags_6 --> scan_intent_tags_9
    scan_intent_tags_8 --> scan_intent_tags_10
    scan_intent_tags_9 --> scan_intent_tags_10
    scan_intent_tags_10 --> scan_intent_tags_11
    scan_intent_tags_11 --> scan_intent_tags_12
```

