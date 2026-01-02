# CFG Group: src/420_dead_code_entrypoints.rs

## Function: `collect_entrypoints`

- File: src/420_dead_code_entrypoints.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    collect_entrypoints_0["ENTRY"]
    collect_entrypoints_1["let mut entrypoints = HashSet :: new ()"]
    collect_entrypoints_2["if let Some (policy) = policy"]
    collect_entrypoints_3["THEN BB"]
    collect_entrypoints_4["for symbol in & policy . entrypoint_symbols { entrypoints . insert (symbol . ..."]
    collect_entrypoints_5["EMPTY ELSE"]
    collect_entrypoints_6["IF JOIN"]
    collect_entrypoints_7["for element in elements { if element . element_type != ElementType :: Functio..."]
    collect_entrypoints_8["entrypoints"]
    collect_entrypoints_9["EXIT"]
    collect_entrypoints_0 --> collect_entrypoints_1
    collect_entrypoints_1 --> collect_entrypoints_2
    collect_entrypoints_2 --> collect_entrypoints_3
    collect_entrypoints_3 --> collect_entrypoints_4
    collect_entrypoints_2 --> collect_entrypoints_5
    collect_entrypoints_4 --> collect_entrypoints_6
    collect_entrypoints_5 --> collect_entrypoints_6
    collect_entrypoints_6 --> collect_entrypoints_7
    collect_entrypoints_7 --> collect_entrypoints_8
    collect_entrypoints_8 --> collect_entrypoints_9
```

## Function: `collect_exports`

- File: src/420_dead_code_entrypoints.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    collect_exports_0["ENTRY"]
    collect_exports_1["let mut exports = HashSet :: new ()"]
    collect_exports_2["let src_dir = root . join ('src')"]
    collect_exports_3["for entry in WalkDir :: new (src_dir) . into_iter () . filter_map (Result :: ..."]
    collect_exports_4["exports"]
    collect_exports_5["EXIT"]
    collect_exports_0 --> collect_exports_1
    collect_exports_1 --> collect_exports_2
    collect_exports_2 --> collect_exports_3
    collect_exports_3 --> collect_exports_4
    collect_exports_4 --> collect_exports_5
```

## Function: `collect_use_tree_idents`

- File: src/420_dead_code_entrypoints.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    collect_use_tree_idents_0["ENTRY"]
    collect_use_tree_idents_1["match tree { syn :: UseTree :: Name (name) => { exports . insert (name . iden..."]
    collect_use_tree_idents_2["EXIT"]
    collect_use_tree_idents_0 --> collect_use_tree_idents_1
    collect_use_tree_idents_1 --> collect_use_tree_idents_2
```

## Function: `is_public_api`

- File: src/420_dead_code_entrypoints.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_public_api_0["ENTRY"]
    is_public_api_1["exports . contains (symbol)"]
    is_public_api_2["EXIT"]
    is_public_api_0 --> is_public_api_1
    is_public_api_1 --> is_public_api_2
```

## Function: `treat_public_as_entrypoint`

- File: src/420_dead_code_entrypoints.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    treat_public_as_entrypoint_0["ENTRY"]
    treat_public_as_entrypoint_1["policy . map (| p | p . treat_public_as_entrypoint) . unwrap_or (true)"]
    treat_public_as_entrypoint_2["EXIT"]
    treat_public_as_entrypoint_0 --> treat_public_as_entrypoint_1
    treat_public_as_entrypoint_1 --> treat_public_as_entrypoint_2
```

