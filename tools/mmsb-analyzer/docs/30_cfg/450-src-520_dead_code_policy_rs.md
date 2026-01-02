# CFG Group: src/520_dead_code_policy.rs

## Function: `load_policy`

- File: src/520_dead_code_policy.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    load_policy_0["ENTRY"]
    load_policy_1["let contents = std :: fs :: read_to_string (path) ?"]
    load_policy_2["Ok (parse_policy (& contents , path . parent () . unwrap_or (path)))"]
    load_policy_3["EXIT"]
    load_policy_0 --> load_policy_1
    load_policy_1 --> load_policy_2
    load_policy_2 --> load_policy_3
```

## Function: `parse_bool`

- File: src/520_dead_code_policy.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    parse_bool_0["ENTRY"]
    parse_bool_1["match value . trim () . to_ascii_lowercase () . as_str () { 'true' => Some (t..."]
    parse_bool_2["EXIT"]
    parse_bool_0 --> parse_bool_1
    parse_bool_1 --> parse_bool_2
```

## Function: `parse_list`

- File: src/520_dead_code_policy.rs
- Branches: 2
- Loops: 0
- Nodes: 14
- Edges: 15

```mermaid
flowchart TD
    parse_list_0["ENTRY"]
    parse_list_1["let mut trimmed = value . trim () . to_string ()"]
    parse_list_2["if let Some (stripped) = trimmed . strip_prefix ('[')"]
    parse_list_3["THEN BB"]
    parse_list_4["trimmed = stripped . to_string ()"]
    parse_list_5["EMPTY ELSE"]
    parse_list_6["IF JOIN"]
    parse_list_7["if let Some (stripped) = trimmed . strip_suffix (']')"]
    parse_list_8["THEN BB"]
    parse_list_9["trimmed = stripped . to_string ()"]
    parse_list_10["EMPTY ELSE"]
    parse_list_11["IF JOIN"]
    parse_list_12["trimmed . split (',') . map (| s | s . trim () . trim_matches (''') . trim_ma..."]
    parse_list_13["EXIT"]
    parse_list_0 --> parse_list_1
    parse_list_1 --> parse_list_2
    parse_list_2 --> parse_list_3
    parse_list_3 --> parse_list_4
    parse_list_2 --> parse_list_5
    parse_list_4 --> parse_list_6
    parse_list_5 --> parse_list_6
    parse_list_6 --> parse_list_7
    parse_list_7 --> parse_list_8
    parse_list_8 --> parse_list_9
    parse_list_7 --> parse_list_10
    parse_list_9 --> parse_list_11
    parse_list_10 --> parse_list_11
    parse_list_11 --> parse_list_12
    parse_list_12 --> parse_list_13
```

## Function: `parse_policy`

- File: src/520_dead_code_policy.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    parse_policy_0["ENTRY"]
    parse_policy_1["let mut planned_directories = Vec :: new ()"]
    parse_policy_2["let mut public_api_roots = Vec :: new ()"]
    parse_policy_3["let mut entrypoint_symbols = Vec :: new ()"]
    parse_policy_4["let mut treat_public_as_entrypoint = true"]
    parse_policy_5["for line in contents . lines () { let trimmed = line . trim () ; if trimmed ...."]
    parse_policy_6["DeadCodePolicy { planned_directories , public_api_roots , entrypoint_symbols ..."]
    parse_policy_7["EXIT"]
    parse_policy_0 --> parse_policy_1
    parse_policy_1 --> parse_policy_2
    parse_policy_2 --> parse_policy_3
    parse_policy_3 --> parse_policy_4
    parse_policy_4 --> parse_policy_5
    parse_policy_5 --> parse_policy_6
    parse_policy_6 --> parse_policy_7
```

