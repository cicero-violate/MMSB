# CFG Group: src/380_dead_code_doc_comment_parser.rs

## Function: `detect_latent_markers`

- File: src/380_dead_code_doc_comment_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    detect_latent_markers_0["ENTRY"]
    detect_latent_markers_1["IntentMarker :: from_comment (comment)"]
    detect_latent_markers_2["EXIT"]
    detect_latent_markers_0 --> detect_latent_markers_1
    detect_latent_markers_1 --> detect_latent_markers_2
```

## Function: `extract_doc_markers`

- File: src/380_dead_code_doc_comment_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    extract_doc_markers_0["ENTRY"]
    extract_doc_markers_1["let mut markers = Vec :: new ()"]
    extract_doc_markers_2["for attr in attrs { if ! attr . path () . is_ident ('doc') { continue ; } let..."]
    extract_doc_markers_3["markers"]
    extract_doc_markers_4["EXIT"]
    extract_doc_markers_0 --> extract_doc_markers_1
    extract_doc_markers_1 --> extract_doc_markers_2
    extract_doc_markers_2 --> extract_doc_markers_3
    extract_doc_markers_3 --> extract_doc_markers_4
```

## Function: `item_attrs`

- File: src/380_dead_code_doc_comment_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    item_attrs_0["ENTRY"]
    item_attrs_1["match item { Item :: Fn (item_fn) => & item_fn . attrs , Item :: Struct (item..."]
    item_attrs_2["EXIT"]
    item_attrs_0 --> item_attrs_1
    item_attrs_1 --> item_attrs_2
```

## Function: `item_name`

- File: src/380_dead_code_doc_comment_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    item_name_0["ENTRY"]
    item_name_1["match item { Item :: Fn (item_fn) => Some (item_fn . sig . ident . to_string ..."]
    item_name_2["EXIT"]
    item_name_0 --> item_name_1
    item_name_1 --> item_name_2
```

## Function: `merge_doc_intent`

- File: src/380_dead_code_doc_comment_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    merge_doc_intent_0["ENTRY"]
    merge_doc_intent_1["let mut merged = IntentMap :: new ()"]
    merge_doc_intent_2["for (symbol , markers) in map { let mut uniques = HashSet :: new () ; for mar..."]
    merge_doc_intent_3["merged"]
    merge_doc_intent_4["EXIT"]
    merge_doc_intent_0 --> merge_doc_intent_1
    merge_doc_intent_1 --> merge_doc_intent_2
    merge_doc_intent_2 --> merge_doc_intent_3
    merge_doc_intent_3 --> merge_doc_intent_4
```

