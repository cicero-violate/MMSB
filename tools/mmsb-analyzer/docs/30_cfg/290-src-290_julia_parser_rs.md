# CFG Group: src/290_julia_parser.rs

## Function: `extract_calls_from_lines`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    extract_calls_from_lines_0["ENTRY"]
    extract_calls_from_lines_1["let joined = lines . join ('\n')"]
    extract_calls_from_lines_2["extract_calls_from_text (& joined)"]
    extract_calls_from_lines_3["EXIT"]
    extract_calls_from_lines_0 --> extract_calls_from_lines_1
    extract_calls_from_lines_1 --> extract_calls_from_lines_2
    extract_calls_from_lines_2 --> extract_calls_from_lines_3
```

## Function: `extract_calls_from_text`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    extract_calls_from_text_0["ENTRY"]
    extract_calls_from_text_1["let mut calls = Vec :: new ()"]
    extract_calls_from_text_2["for capture in CALL_RE . captures_iter (text) { if let Some (name) = capture ..."]
    extract_calls_from_text_3["calls . sort ()"]
    extract_calls_from_text_4["calls . dedup ()"]
    extract_calls_from_text_5["calls"]
    extract_calls_from_text_6["EXIT"]
    extract_calls_from_text_0 --> extract_calls_from_text_1
    extract_calls_from_text_1 --> extract_calls_from_text_2
    extract_calls_from_text_2 --> extract_calls_from_text_3
    extract_calls_from_text_3 --> extract_calls_from_text_4
    extract_calls_from_text_4 --> extract_calls_from_text_5
    extract_calls_from_text_5 --> extract_calls_from_text_6
```

## Function: `find_julia_project_dir`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    find_julia_project_dir_0["ENTRY"]
    find_julia_project_dir_1["let mut current = script_path . parent ()"]
    find_julia_project_dir_2["while let Some (dir) = current { if dir . join ('Project.toml') . exists () {..."]
    find_julia_project_dir_3["script_path . parent () . unwrap_or_else (| | Path :: new ('.')) . to_path_bu..."]
    find_julia_project_dir_4["EXIT"]
    find_julia_project_dir_0 --> find_julia_project_dir_1
    find_julia_project_dir_1 --> find_julia_project_dir_2
    find_julia_project_dir_2 --> find_julia_project_dir_3
    find_julia_project_dir_3 --> find_julia_project_dir_4
```

## Function: `is_reserved`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_reserved_0["ENTRY"]
    is_reserved_1["matches ! (name , 'if' | 'for' | 'while' | 'begin' | 'let' | 'struct' | 'muta..."]
    is_reserved_2["EXIT"]
    is_reserved_0 --> is_reserved_1
    is_reserved_1 --> is_reserved_2
```

## Function: `paren_balance`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    paren_balance_0["ENTRY"]
    paren_balance_1["let mut balance = 0i32"]
    paren_balance_2["for ch in input . chars () { if ch == '(' { balance += 1 ; } else if ch == ')..."]
    paren_balance_3["balance"]
    paren_balance_4["EXIT"]
    paren_balance_0 --> paren_balance_1
    paren_balance_1 --> paren_balance_2
    paren_balance_2 --> paren_balance_3
    paren_balance_3 --> paren_balance_4
```

## Function: `parse_module_name`

- File: src/290_julia_parser.rs
- Branches: 2
- Loops: 0
- Nodes: 13
- Edges: 14

```mermaid
flowchart TD
    parse_module_name_0["ENTRY"]
    parse_module_name_1["if line . starts_with ('module ')"]
    parse_module_name_2["THEN BB"]
    parse_module_name_3["return line . split_whitespace () . nth (1) . map (| name | name . trim_end_m..."]
    parse_module_name_4["EMPTY ELSE"]
    parse_module_name_5["IF JOIN"]
    parse_module_name_6["if line . starts_with ('baremodule ')"]
    parse_module_name_7["THEN BB"]
    parse_module_name_8["return line . split_whitespace () . nth (1) . map (| name | name . trim_end_m..."]
    parse_module_name_9["EMPTY ELSE"]
    parse_module_name_10["IF JOIN"]
    parse_module_name_11["None"]
    parse_module_name_12["EXIT"]
    parse_module_name_0 --> parse_module_name_1
    parse_module_name_1 --> parse_module_name_2
    parse_module_name_2 --> parse_module_name_3
    parse_module_name_1 --> parse_module_name_4
    parse_module_name_3 --> parse_module_name_5
    parse_module_name_4 --> parse_module_name_5
    parse_module_name_5 --> parse_module_name_6
    parse_module_name_6 --> parse_module_name_7
    parse_module_name_7 --> parse_module_name_8
    parse_module_name_6 --> parse_module_name_9
    parse_module_name_8 --> parse_module_name_10
    parse_module_name_9 --> parse_module_name_10
    parse_module_name_10 --> parse_module_name_11
    parse_module_name_11 --> parse_module_name_12
```

## Function: `parse_struct_name`

- File: src/290_julia_parser.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    parse_struct_name_0["ENTRY"]
    parse_struct_name_1["if line . starts_with ('mutable struct ') || line . starts_with ('struct ')"]
    parse_struct_name_2["THEN BB"]
    parse_struct_name_3["let offset = if line . starts_with ('mutable struct ') { 2 } else { 1 }"]
    parse_struct_name_4["let tokens : Vec < & str > = line . split_whitespace () . collect ()"]
    parse_struct_name_5["return tokens . get (offset) . map (| name | { name . split ('<:') . next () ..."]
    parse_struct_name_6["EMPTY ELSE"]
    parse_struct_name_7["IF JOIN"]
    parse_struct_name_8["None"]
    parse_struct_name_9["EXIT"]
    parse_struct_name_0 --> parse_struct_name_1
    parse_struct_name_1 --> parse_struct_name_2
    parse_struct_name_2 --> parse_struct_name_3
    parse_struct_name_3 --> parse_struct_name_4
    parse_struct_name_4 --> parse_struct_name_5
    parse_struct_name_1 --> parse_struct_name_6
    parse_struct_name_5 --> parse_struct_name_7
    parse_struct_name_6 --> parse_struct_name_7
    parse_struct_name_7 --> parse_struct_name_8
    parse_struct_name_8 --> parse_struct_name_9
```

## Function: `relativize_path`

- File: src/290_julia_parser.rs
- Branches: 1
- Loops: 0
- Nodes: 8
- Edges: 8

```mermaid
flowchart TD
    relativize_path_0["ENTRY"]
    relativize_path_1["if let Ok (stripped) = path . strip_prefix (root)"]
    relativize_path_2["THEN BB"]
    relativize_path_3["stripped . to_string_lossy () . to_string ()"]
    relativize_path_4["ELSE BB"]
    relativize_path_5["{ path . to_string_lossy () . to_string () }"]
    relativize_path_6["IF JOIN"]
    relativize_path_7["EXIT"]
    relativize_path_0 --> relativize_path_1
    relativize_path_1 --> relativize_path_2
    relativize_path_2 --> relativize_path_3
    relativize_path_1 --> relativize_path_4
    relativize_path_4 --> relativize_path_5
    relativize_path_3 --> relativize_path_6
    relativize_path_5 --> relativize_path_6
    relativize_path_6 --> relativize_path_7
```

## Function: `resolve_julia_binary`

- File: src/290_julia_parser.rs
- Branches: 5
- Loops: 0
- Nodes: 29
- Edges: 33

```mermaid
flowchart TD
    resolve_julia_binary_0["ENTRY"]
    resolve_julia_binary_1["if let Ok (custom) = env :: var ('JULIA_BINARY')"]
    resolve_julia_binary_2["THEN BB"]
    resolve_julia_binary_3["let candidate = PathBuf :: from (custom)"]
    resolve_julia_binary_4["if candidate . exists ()"]
    resolve_julia_binary_5["THEN BB"]
    resolve_julia_binary_6["return candidate"]
    resolve_julia_binary_7["EMPTY ELSE"]
    resolve_julia_binary_8["IF JOIN"]
    resolve_julia_binary_9["EMPTY ELSE"]
    resolve_julia_binary_10["IF JOIN"]
    resolve_julia_binary_11["if let Ok (home) = env :: var ('HOME')"]
    resolve_julia_binary_12["THEN BB"]
    resolve_julia_binary_13["let juliaup_root = Path :: new (& home) . join ('.julia/juliaup')"]
    resolve_julia_binary_14["if let Ok (entries) = fs :: read_dir (& juliaup_root)"]
    resolve_julia_binary_15["THEN BB"]
    resolve_julia_binary_16["for entry in entries . flatten () { let path = entry . path () ; if ! path . ..."]
    resolve_julia_binary_17["EMPTY ELSE"]
    resolve_julia_binary_18["IF JOIN"]
    resolve_julia_binary_19["EMPTY ELSE"]
    resolve_julia_binary_20["IF JOIN"]
    resolve_julia_binary_21["let alt = PathBuf :: from ('/home/cicero-arch-omen/git/julia/usr/bin/julia')"]
    resolve_julia_binary_22["if alt . exists ()"]
    resolve_julia_binary_23["THEN BB"]
    resolve_julia_binary_24["return alt"]
    resolve_julia_binary_25["EMPTY ELSE"]
    resolve_julia_binary_26["IF JOIN"]
    resolve_julia_binary_27["PathBuf :: from ('julia')"]
    resolve_julia_binary_28["EXIT"]
    resolve_julia_binary_0 --> resolve_julia_binary_1
    resolve_julia_binary_1 --> resolve_julia_binary_2
    resolve_julia_binary_2 --> resolve_julia_binary_3
    resolve_julia_binary_3 --> resolve_julia_binary_4
    resolve_julia_binary_4 --> resolve_julia_binary_5
    resolve_julia_binary_5 --> resolve_julia_binary_6
    resolve_julia_binary_4 --> resolve_julia_binary_7
    resolve_julia_binary_6 --> resolve_julia_binary_8
    resolve_julia_binary_7 --> resolve_julia_binary_8
    resolve_julia_binary_1 --> resolve_julia_binary_9
    resolve_julia_binary_8 --> resolve_julia_binary_10
    resolve_julia_binary_9 --> resolve_julia_binary_10
    resolve_julia_binary_10 --> resolve_julia_binary_11
    resolve_julia_binary_11 --> resolve_julia_binary_12
    resolve_julia_binary_12 --> resolve_julia_binary_13
    resolve_julia_binary_13 --> resolve_julia_binary_14
    resolve_julia_binary_14 --> resolve_julia_binary_15
    resolve_julia_binary_15 --> resolve_julia_binary_16
    resolve_julia_binary_14 --> resolve_julia_binary_17
    resolve_julia_binary_16 --> resolve_julia_binary_18
    resolve_julia_binary_17 --> resolve_julia_binary_18
    resolve_julia_binary_11 --> resolve_julia_binary_19
    resolve_julia_binary_18 --> resolve_julia_binary_20
    resolve_julia_binary_19 --> resolve_julia_binary_20
    resolve_julia_binary_20 --> resolve_julia_binary_21
    resolve_julia_binary_21 --> resolve_julia_binary_22
    resolve_julia_binary_22 --> resolve_julia_binary_23
    resolve_julia_binary_23 --> resolve_julia_binary_24
    resolve_julia_binary_22 --> resolve_julia_binary_25
    resolve_julia_binary_24 --> resolve_julia_binary_26
    resolve_julia_binary_25 --> resolve_julia_binary_26
    resolve_julia_binary_26 --> resolve_julia_binary_27
    resolve_julia_binary_27 --> resolve_julia_binary_28
```

## Function: `slugify_relative`

- File: src/290_julia_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    slugify_relative_0["ENTRY"]
    slugify_relative_1["let relative = path . strip_prefix (root) . unwrap_or (path)"]
    slugify_relative_2["relative . components () . map (| c | c . as_os_str () . to_string_lossy () ...."]
    slugify_relative_3["EXIT"]
    slugify_relative_0 --> slugify_relative_1
    slugify_relative_1 --> slugify_relative_2
    slugify_relative_2 --> slugify_relative_3
```

