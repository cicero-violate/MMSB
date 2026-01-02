# CFG Group: src/260_directory_analyzer.rs

## Function: `is_source_file`

- File: src/260_directory_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_source_file_0["ENTRY"]
    is_source_file_1["matches ! (path . extension () . and_then (| e | e . to_str ()) , Some ('rs')..."]
    is_source_file_2["EXIT"]
    is_source_file_0 --> is_source_file_1
    is_source_file_1 --> is_source_file_2
```

## Function: `should_skip_path`

- File: src/260_directory_analyzer.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    should_skip_path_0["ENTRY"]
    should_skip_path_1["let Some (name) = path . file_name () else { return false ; }"]
    should_skip_path_2["name == 'target' || name == '.git' || name == 'tools' || name == 'examples' |..."]
    should_skip_path_3["EXIT"]
    should_skip_path_0 --> should_skip_path_1
    should_skip_path_1 --> should_skip_path_2
    should_skip_path_2 --> should_skip_path_3
```

