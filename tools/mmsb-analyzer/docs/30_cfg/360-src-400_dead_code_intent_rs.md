# CFG Group: src/400_dead_code_intent.rs

## Function: `check_planned_directory`

- File: src/400_dead_code_intent.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    check_planned_directory_0["ENTRY"]
    check_planned_directory_1["let Some (policy) = policy else { return false ; }"]
    check_planned_directory_2["for dir in & policy . planned_directories { if path . starts_with (dir) { ret..."]
    check_planned_directory_3["false"]
    check_planned_directory_4["EXIT"]
    check_planned_directory_0 --> check_planned_directory_1
    check_planned_directory_1 --> check_planned_directory_2
    check_planned_directory_2 --> check_planned_directory_3
    check_planned_directory_3 --> check_planned_directory_4
```

## Function: `collect_symbols`

- File: src/400_dead_code_intent.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    collect_symbols_0["ENTRY"]
    collect_symbols_1["let contents = std :: fs :: read_to_string (file) . unwrap_or_default ()"]
    collect_symbols_2["let parsed = match syn :: parse_file (& contents) { Ok (file) => file , Err (_) => return ..."]
    collect_symbols_3["parsed . items . iter () . filter_map (item_name) . collect ()"]
    collect_symbols_4["EXIT"]
    collect_symbols_0 --> collect_symbols_1
    collect_symbols_1 --> collect_symbols_2
    collect_symbols_2 --> collect_symbols_3
    collect_symbols_3 --> collect_symbols_4
```

## Function: `merge_intent_sources`

- File: src/400_dead_code_intent.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    merge_intent_sources_0["ENTRY"]
    merge_intent_sources_1["let mut merged = IntentMap :: new ()"]
    merge_intent_sources_2["for (symbol , items) in attrs { merged . entry (symbol) . or_default () . ext..."]
    merge_intent_sources_3["for (symbol , items) in docs { merged . entry (symbol) . or_default () . exte..."]
    merge_intent_sources_4["for (symbol , items) in dir { merged . entry (symbol) . or_default () . exten..."]
    merge_intent_sources_5["merged"]
    merge_intent_sources_6["EXIT"]
    merge_intent_sources_0 --> merge_intent_sources_1
    merge_intent_sources_1 --> merge_intent_sources_2
    merge_intent_sources_2 --> merge_intent_sources_3
    merge_intent_sources_3 --> merge_intent_sources_4
    merge_intent_sources_4 --> merge_intent_sources_5
    merge_intent_sources_5 --> merge_intent_sources_6
```

## Function: `planned_directory_intent`

- File: src/400_dead_code_intent.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    planned_directory_intent_0["ENTRY"]
    planned_directory_intent_1["if ! check_planned_directory (file , policy)"]
    planned_directory_intent_2["THEN BB"]
    planned_directory_intent_3["return IntentMap :: new ()"]
    planned_directory_intent_4["EMPTY ELSE"]
    planned_directory_intent_5["IF JOIN"]
    planned_directory_intent_6["let mut map : IntentMap = HashMap :: new ()"]
    planned_directory_intent_7["for symbol in collect_symbols (file) { map . entry (symbol) . or_default () ...."]
    planned_directory_intent_8["map"]
    planned_directory_intent_9["EXIT"]
    planned_directory_intent_0 --> planned_directory_intent_1
    planned_directory_intent_1 --> planned_directory_intent_2
    planned_directory_intent_2 --> planned_directory_intent_3
    planned_directory_intent_1 --> planned_directory_intent_4
    planned_directory_intent_3 --> planned_directory_intent_5
    planned_directory_intent_4 --> planned_directory_intent_5
    planned_directory_intent_5 --> planned_directory_intent_6
    planned_directory_intent_6 --> planned_directory_intent_7
    planned_directory_intent_7 --> planned_directory_intent_8
    planned_directory_intent_8 --> planned_directory_intent_9
```

