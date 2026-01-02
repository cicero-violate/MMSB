# CFG Group: src/300_rust_parser.rs

## Function: `expr_snippet`

- File: src/300_rust_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    expr_snippet_0["ENTRY"]
    expr_snippet_1["truncate_label (quote :: quote ! (# expr) . to_string ())"]
    expr_snippet_2["EXIT"]
    expr_snippet_0 --> expr_snippet_1
    expr_snippet_1 --> expr_snippet_2
```

## Function: `pat_snippet`

- File: src/300_rust_parser.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    pat_snippet_0["ENTRY"]
    pat_snippet_1["truncate_label (quote :: quote ! (# pat) . to_string ())"]
    pat_snippet_2["EXIT"]
    pat_snippet_0 --> pat_snippet_1
    pat_snippet_1 --> pat_snippet_2
```

## Function: `relativize_path`

- File: src/300_rust_parser.rs
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

## Function: `truncate_label`

- File: src/300_rust_parser.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    truncate_label_0["ENTRY"]
    truncate_label_1["let collapsed = text . split_whitespace () . collect :: < Vec < _ > > () . join (' ')"]
    truncate_label_2["let mut label = collapsed"]
    truncate_label_3["if label . len () > 80"]
    truncate_label_4["THEN BB"]
    truncate_label_5["label . truncate (77)"]
    truncate_label_6["label . push_str ('...')"]
    truncate_label_7["EMPTY ELSE"]
    truncate_label_8["IF JOIN"]
    truncate_label_9["label"]
    truncate_label_10["EXIT"]
    truncate_label_0 --> truncate_label_1
    truncate_label_1 --> truncate_label_2
    truncate_label_2 --> truncate_label_3
    truncate_label_3 --> truncate_label_4
    truncate_label_4 --> truncate_label_5
    truncate_label_5 --> truncate_label_6
    truncate_label_3 --> truncate_label_7
    truncate_label_6 --> truncate_label_8
    truncate_label_7 --> truncate_label_8
    truncate_label_8 --> truncate_label_9
    truncate_label_9 --> truncate_label_10
```

