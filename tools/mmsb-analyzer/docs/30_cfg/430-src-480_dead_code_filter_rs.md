# CFG Group: src/480_dead_code_filter.rs

## Function: `collect_excluded_symbols`

- File: src/480_dead_code_filter.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    collect_excluded_symbols_0["ENTRY"]
    collect_excluded_symbols_1["report . items . iter () . filter (| item | should_exclude_from_analysis (ite..."]
    collect_excluded_symbols_2["EXIT"]
    collect_excluded_symbols_0 --> collect_excluded_symbols_1
    collect_excluded_symbols_1 --> collect_excluded_symbols_2
```

## Function: `filter_dead_code_elements`

- File: src/480_dead_code_filter.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    filter_dead_code_elements_0["ENTRY"]
    filter_dead_code_elements_1["let excluded = collect_excluded_symbols (report)"]
    filter_dead_code_elements_2["elements . iter () . filter (| el | ! excluded . contains (& el . name)) . cl..."]
    filter_dead_code_elements_3["EXIT"]
    filter_dead_code_elements_0 --> filter_dead_code_elements_1
    filter_dead_code_elements_1 --> filter_dead_code_elements_2
    filter_dead_code_elements_2 --> filter_dead_code_elements_3
```

## Function: `should_exclude_from_analysis`

- File: src/480_dead_code_filter.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    should_exclude_from_analysis_0["ENTRY"]
    should_exclude_from_analysis_1["matches ! (category , DeadCodeCategory :: Unreachable | DeadCodeCategory :: T..."]
    should_exclude_from_analysis_2["EXIT"]
    should_exclude_from_analysis_0 --> should_exclude_from_analysis_1
    should_exclude_from_analysis_1 --> should_exclude_from_analysis_2
```

