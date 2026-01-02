# CFG Group: src/350_agent_cli.rs

## Function: `check_action`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    check_action_0["ENTRY"]
    check_action_1["let action_json = std :: fs :: read_to_string (action_path) ?"]
    check_action_2["let action : AgentAction = serde_json :: from_str (& action_json) ?"]
    check_action_3["let invariants = load_invariants (conscience_path) ?"]
    check_action_4["let conscience = AgentConscience :: new (invariants)"]
    check_action_5["let result = conscience . check_action (& action)"]
    check_action_6["let output = serde_json :: to_string_pretty (& result) ?"]
    check_action_7["macro println"]
    check_action_8["std :: process :: exit (if result . allowed { 0 } else { 1 })"]
    check_action_9["EXIT"]
    check_action_0 --> check_action_1
    check_action_1 --> check_action_2
    check_action_2 --> check_action_3
    check_action_3 --> check_action_4
    check_action_4 --> check_action_5
    check_action_5 --> check_action_6
    check_action_6 --> check_action_7
    check_action_7 --> check_action_8
    check_action_8 --> check_action_9
```

## Function: `list_invariants`

- File: src/350_agent_cli.rs
- Branches: 1
- Loops: 0
- Nodes: 13
- Edges: 13

```mermaid
flowchart TD
    list_invariants_0["ENTRY"]
    list_invariants_1["let invariants = load_invariants (conscience_path) ?"]
    list_invariants_2["let filtered : Vec < _ > = if blocking_only { invariants . iter () . filter (| i | i . is_blocking ()) ...."]
    list_invariants_3["macro println"]
    list_invariants_4["if blocking_only"]
    list_invariants_5["THEN BB"]
    list_invariants_6["macro println"]
    list_invariants_7["EMPTY ELSE"]
    list_invariants_8["IF JOIN"]
    list_invariants_9["let output = serde_json :: to_string_pretty (& filtered) ?"]
    list_invariants_10["macro println"]
    list_invariants_11["Ok (())"]
    list_invariants_12["EXIT"]
    list_invariants_0 --> list_invariants_1
    list_invariants_1 --> list_invariants_2
    list_invariants_2 --> list_invariants_3
    list_invariants_3 --> list_invariants_4
    list_invariants_4 --> list_invariants_5
    list_invariants_5 --> list_invariants_6
    list_invariants_4 --> list_invariants_7
    list_invariants_6 --> list_invariants_8
    list_invariants_7 --> list_invariants_8
    list_invariants_8 --> list_invariants_9
    list_invariants_9 --> list_invariants_10
    list_invariants_10 --> list_invariants_11
    list_invariants_11 --> list_invariants_12
```

## Function: `load_invariants`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    load_invariants_0["ENTRY"]
    load_invariants_1["let json = std :: fs :: read_to_string (path) ?"]
    load_invariants_2["Ok (serde_json :: from_str (& json) ?)"]
    load_invariants_3["EXIT"]
    load_invariants_0 --> load_invariants_1
    load_invariants_1 --> load_invariants_2
    load_invariants_2 --> load_invariants_3
```

## Function: `query_function`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    query_function_0["ENTRY"]
    query_function_1["let invariants = load_invariants (conscience_path) ?"]
    query_function_2["let conscience = AgentConscience :: new (invariants)"]
    query_function_3["let allowed = conscience . query_allowed_actions (function)"]
    query_function_4["let output = serde_json :: to_string_pretty (& allowed) ?"]
    query_function_5["macro println"]
    query_function_6["Ok (())"]
    query_function_7["EXIT"]
    query_function_0 --> query_function_1
    query_function_1 --> query_function_2
    query_function_2 --> query_function_3
    query_function_3 --> query_function_4
    query_function_4 --> query_function_5
    query_function_5 --> query_function_6
    query_function_6 --> query_function_7
```

## Function: `run_agent_cli`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    run_agent_cli_0["ENTRY"]
    run_agent_cli_1["let cli = AgentCli :: parse ()"]
    run_agent_cli_2["match cli . command { AgentCommand :: Check { action , conscience } => { chec..."]
    run_agent_cli_3["Ok (())"]
    run_agent_cli_4["EXIT"]
    run_agent_cli_0 --> run_agent_cli_1
    run_agent_cli_1 --> run_agent_cli_2
    run_agent_cli_2 --> run_agent_cli_3
    run_agent_cli_3 --> run_agent_cli_4
```

## Function: `show_stats`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    show_stats_0["ENTRY"]
    show_stats_1["let invariants = load_invariants (conscience_path) ?"]
    show_stats_2["let conscience = AgentConscience :: new (invariants)"]
    show_stats_3["let stats = conscience . stats ()"]
    show_stats_4["macro println"]
    show_stats_5["macro println"]
    show_stats_6["macro println"]
    show_stats_7["macro println"]
    show_stats_8["macro println"]
    show_stats_9["macro println"]
    show_stats_10["macro println"]
    show_stats_11["macro println"]
    show_stats_12["macro println"]
    show_stats_13["macro println"]
    show_stats_14["Ok (())"]
    show_stats_15["EXIT"]
    show_stats_0 --> show_stats_1
    show_stats_1 --> show_stats_2
    show_stats_2 --> show_stats_3
    show_stats_3 --> show_stats_4
    show_stats_4 --> show_stats_5
    show_stats_5 --> show_stats_6
    show_stats_6 --> show_stats_7
    show_stats_7 --> show_stats_8
    show_stats_8 --> show_stats_9
    show_stats_9 --> show_stats_10
    show_stats_10 --> show_stats_11
    show_stats_11 --> show_stats_12
    show_stats_12 --> show_stats_13
    show_stats_13 --> show_stats_14
    show_stats_14 --> show_stats_15
```

## Function: `test_load_invariants_empty`

- File: src/350_agent_cli.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_load_invariants_empty_0["ENTRY"]
    test_load_invariants_empty_1["let temp_path = std :: env :: temp_dir () . join ('test_invariants.json')"]
    test_load_invariants_empty_2["std :: fs :: write (& temp_path , '[]') . unwrap ()"]
    test_load_invariants_empty_3["let result = load_invariants (& temp_path)"]
    test_load_invariants_empty_4["macro assert"]
    test_load_invariants_empty_5["macro assert_eq"]
    test_load_invariants_empty_6["let _ = std :: fs :: remove_file (& temp_path)"]
    test_load_invariants_empty_7["EXIT"]
    test_load_invariants_empty_0 --> test_load_invariants_empty_1
    test_load_invariants_empty_1 --> test_load_invariants_empty_2
    test_load_invariants_empty_2 --> test_load_invariants_empty_3
    test_load_invariants_empty_3 --> test_load_invariants_empty_4
    test_load_invariants_empty_4 --> test_load_invariants_empty_5
    test_load_invariants_empty_5 --> test_load_invariants_empty_6
    test_load_invariants_empty_6 --> test_load_invariants_empty_7
```

