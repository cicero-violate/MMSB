# CFG Group: src/540_dead_code_report_split.rs

## Function: `plan_options`

- File: src/540_dead_code_report_split.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    plan_options_0["ENTRY"]
    plan_options_1["let options = match item . category { DeadCodeCategory :: Unreachable => 'keep | quarantine..."]
    plan_options_2["let mut plan = format ! ('review_only; options: {}' , options)"]
    plan_options_3["if item . action == RecommendedAction :: DeleteSafe"]
    plan_options_4["THEN BB"]
    plan_options_5["plan . push_str ('; requires dead_code warning')"]
    plan_options_6["EMPTY ELSE"]
    plan_options_7["IF JOIN"]
    plan_options_8["plan"]
    plan_options_9["EXIT"]
    plan_options_0 --> plan_options_1
    plan_options_1 --> plan_options_2
    plan_options_2 --> plan_options_3
    plan_options_3 --> plan_options_4
    plan_options_4 --> plan_options_5
    plan_options_3 --> plan_options_6
    plan_options_5 --> plan_options_7
    plan_options_6 --> plan_options_7
    plan_options_7 --> plan_options_8
    plan_options_8 --> plan_options_9
```

## Function: `top_items`

- File: src/540_dead_code_report_split.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    top_items_0["ENTRY"]
    top_items_1["let mut items = items . to_vec ()"]
    top_items_2["items . sort_by_key (| item | item . action as u8)"]
    top_items_3["if items . len () > limit"]
    top_items_4["THEN BB"]
    top_items_5["items . truncate (limit)"]
    top_items_6["EMPTY ELSE"]
    top_items_7["IF JOIN"]
    top_items_8["items"]
    top_items_9["EXIT"]
    top_items_0 --> top_items_1
    top_items_1 --> top_items_2
    top_items_2 --> top_items_3
    top_items_3 --> top_items_4
    top_items_4 --> top_items_5
    top_items_3 --> top_items_6
    top_items_5 --> top_items_7
    top_items_6 --> top_items_7
    top_items_7 --> top_items_8
    top_items_8 --> top_items_9
```

## Function: `write_plan_markdown`

- File: src/540_dead_code_report_split.rs
- Branches: 1
- Loops: 0
- Nodes: 17
- Edges: 17

```mermaid
flowchart TD
    write_plan_markdown_0["ENTRY"]
    write_plan_markdown_1["let mut content = String :: new ()"]
    write_plan_markdown_2["content . push_str ('# Dead Code Plans (Review Only)\n\n')"]
    write_plan_markdown_3["content . push_str (& format ! ('Generated: {}\n\n' , report . timestamp))"]
    write_plan_markdown_4["content . push_str ('Policy: review_only. No automatic deletion or moves.\n')"]
    write_plan_markdown_5["content . push_str ('Guards: never delete public API; delete_safe requires ma..."]
    write_plan_markdown_6["let items = top_items (& report . items , limit)"]
    write_plan_markdown_7["content . push_str ('## Planned Items\n\n')"]
    write_plan_markdown_8["if items . is_empty ()"]
    write_plan_markdown_9["THEN BB"]
    write_plan_markdown_10["content . push_str ('- None.\n')"]
    write_plan_markdown_11["ELSE BB"]
    write_plan_markdown_12["{ for item in items { let plan = plan_options (& item) ; content . push_str (..."]
    write_plan_markdown_13["IF JOIN"]
    write_plan_markdown_14["content . push ('\n')"]
    write_plan_markdown_15["std :: fs :: write (path , content)"]
    write_plan_markdown_16["EXIT"]
    write_plan_markdown_0 --> write_plan_markdown_1
    write_plan_markdown_1 --> write_plan_markdown_2
    write_plan_markdown_2 --> write_plan_markdown_3
    write_plan_markdown_3 --> write_plan_markdown_4
    write_plan_markdown_4 --> write_plan_markdown_5
    write_plan_markdown_5 --> write_plan_markdown_6
    write_plan_markdown_6 --> write_plan_markdown_7
    write_plan_markdown_7 --> write_plan_markdown_8
    write_plan_markdown_8 --> write_plan_markdown_9
    write_plan_markdown_9 --> write_plan_markdown_10
    write_plan_markdown_8 --> write_plan_markdown_11
    write_plan_markdown_11 --> write_plan_markdown_12
    write_plan_markdown_10 --> write_plan_markdown_13
    write_plan_markdown_12 --> write_plan_markdown_13
    write_plan_markdown_13 --> write_plan_markdown_14
    write_plan_markdown_14 --> write_plan_markdown_15
    write_plan_markdown_15 --> write_plan_markdown_16
```

## Function: `write_summary_markdown`

- File: src/540_dead_code_report_split.rs
- Branches: 1
- Loops: 0
- Nodes: 17
- Edges: 17

```mermaid
flowchart TD
    write_summary_markdown_0["ENTRY"]
    write_summary_markdown_1["let mut content = String :: new ()"]
    write_summary_markdown_2["content . push_str ('# Dead Code Summary\n\n')"]
    write_summary_markdown_3["content . push_str (& format ! ('Generated: {}\n\n' , report . timestamp))"]
    write_summary_markdown_4["content . push_str ('## Summary Counts\n\n')"]
    write_summary_markdown_5["content . push_str (& format ! ('- Unreachable: {}\n- Reachable-unused: {}\n-..."]
    write_summary_markdown_6["let items = top_items (& report . items , limit)"]
    write_summary_markdown_7["content . push_str ('## Top Findings\n\n')"]
    write_summary_markdown_8["if items . is_empty ()"]
    write_summary_markdown_9["THEN BB"]
    write_summary_markdown_10["content . push_str ('- None.\n')"]
    write_summary_markdown_11["ELSE BB"]
    write_summary_markdown_12["{ for item in items { content . push_str (& format ! ('- '{}' in '{}' — {:?..."]
    write_summary_markdown_13["IF JOIN"]
    write_summary_markdown_14["content . push ('\n')"]
    write_summary_markdown_15["std :: fs :: write (path , content)"]
    write_summary_markdown_16["EXIT"]
    write_summary_markdown_0 --> write_summary_markdown_1
    write_summary_markdown_1 --> write_summary_markdown_2
    write_summary_markdown_2 --> write_summary_markdown_3
    write_summary_markdown_3 --> write_summary_markdown_4
    write_summary_markdown_4 --> write_summary_markdown_5
    write_summary_markdown_5 --> write_summary_markdown_6
    write_summary_markdown_6 --> write_summary_markdown_7
    write_summary_markdown_7 --> write_summary_markdown_8
    write_summary_markdown_8 --> write_summary_markdown_9
    write_summary_markdown_9 --> write_summary_markdown_10
    write_summary_markdown_8 --> write_summary_markdown_11
    write_summary_markdown_11 --> write_summary_markdown_12
    write_summary_markdown_10 --> write_summary_markdown_13
    write_summary_markdown_12 --> write_summary_markdown_13
    write_summary_markdown_13 --> write_summary_markdown_14
    write_summary_markdown_14 --> write_summary_markdown_15
    write_summary_markdown_15 --> write_summary_markdown_16
```

