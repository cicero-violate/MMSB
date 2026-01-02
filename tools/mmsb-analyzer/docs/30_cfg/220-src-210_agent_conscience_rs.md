# CFG Group: src/210_agent_conscience.rs

## Function: `make_test_invariant`

- File: src/210_agent_conscience.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    make_test_invariant_0["ENTRY"]
    make_test_invariant_1["Invariant :: new (name . to_string () , format ! ('src/{:03}_test.rs' , layer..."]
    make_test_invariant_2["EXIT"]
    make_test_invariant_0 --> make_test_invariant_1
    make_test_invariant_1 --> make_test_invariant_2
```

## Function: `test_conscience_allows_valid_action`

- File: src/210_agent_conscience.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_conscience_allows_valid_action_0["ENTRY"]
    test_conscience_allows_valid_action_1["let inv = make_test_invariant ('other_fn' , 0 , InvariantStrength :: Proven)"]
    test_conscience_allows_valid_action_2["let conscience = AgentConscience :: new (vec ! [inv])"]
    test_conscience_allows_valid_action_3["let action = AgentAction :: MoveFunction { name : 'test_fn' . to_string () , from : PathBu..."]
    test_conscience_allows_valid_action_4["let result = conscience . check_action (& action)"]
    test_conscience_allows_valid_action_5["macro assert"]
    test_conscience_allows_valid_action_6["macro assert"]
    test_conscience_allows_valid_action_7["EXIT"]
    test_conscience_allows_valid_action_0 --> test_conscience_allows_valid_action_1
    test_conscience_allows_valid_action_1 --> test_conscience_allows_valid_action_2
    test_conscience_allows_valid_action_2 --> test_conscience_allows_valid_action_3
    test_conscience_allows_valid_action_3 --> test_conscience_allows_valid_action_4
    test_conscience_allows_valid_action_4 --> test_conscience_allows_valid_action_5
    test_conscience_allows_valid_action_5 --> test_conscience_allows_valid_action_6
    test_conscience_allows_valid_action_6 --> test_conscience_allows_valid_action_7
```

## Function: `test_conscience_blocks_invalid_move`

- File: src/210_agent_conscience.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_conscience_blocks_invalid_move_0["ENTRY"]
    test_conscience_blocks_invalid_move_1["let inv = make_test_invariant ('test_fn' , 0 , InvariantStrength :: Proven)"]
    test_conscience_blocks_invalid_move_2["let conscience = AgentConscience :: new (vec ! [inv])"]
    test_conscience_blocks_invalid_move_3["let action = AgentAction :: MoveFunction { name : 'test_fn' . to_string () , from : PathBu..."]
    test_conscience_blocks_invalid_move_4["let result = conscience . check_action (& action)"]
    test_conscience_blocks_invalid_move_5["macro assert"]
    test_conscience_blocks_invalid_move_6["macro assert"]
    test_conscience_blocks_invalid_move_7["EXIT"]
    test_conscience_blocks_invalid_move_0 --> test_conscience_blocks_invalid_move_1
    test_conscience_blocks_invalid_move_1 --> test_conscience_blocks_invalid_move_2
    test_conscience_blocks_invalid_move_2 --> test_conscience_blocks_invalid_move_3
    test_conscience_blocks_invalid_move_3 --> test_conscience_blocks_invalid_move_4
    test_conscience_blocks_invalid_move_4 --> test_conscience_blocks_invalid_move_5
    test_conscience_blocks_invalid_move_5 --> test_conscience_blocks_invalid_move_6
    test_conscience_blocks_invalid_move_6 --> test_conscience_blocks_invalid_move_7
```

## Function: `test_conscience_stats`

- File: src/210_agent_conscience.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_conscience_stats_0["ENTRY"]
    test_conscience_stats_1["let invariants = vec ! [make_test_invariant ('fn1' , 0 , InvariantStrength :: Proven) , make_t..."]
    test_conscience_stats_2["let conscience = AgentConscience :: new (invariants)"]
    test_conscience_stats_3["let stats = conscience . stats ()"]
    test_conscience_stats_4["macro assert_eq"]
    test_conscience_stats_5["macro assert_eq"]
    test_conscience_stats_6["macro assert_eq"]
    test_conscience_stats_7["macro assert_eq"]
    test_conscience_stats_8["EXIT"]
    test_conscience_stats_0 --> test_conscience_stats_1
    test_conscience_stats_1 --> test_conscience_stats_2
    test_conscience_stats_2 --> test_conscience_stats_3
    test_conscience_stats_3 --> test_conscience_stats_4
    test_conscience_stats_4 --> test_conscience_stats_5
    test_conscience_stats_5 --> test_conscience_stats_6
    test_conscience_stats_6 --> test_conscience_stats_7
    test_conscience_stats_7 --> test_conscience_stats_8
```

## Function: `test_query_allowed_actions`

- File: src/210_agent_conscience.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_query_allowed_actions_0["ENTRY"]
    test_query_allowed_actions_1["let inv = make_test_invariant ('test_fn' , 0 , InvariantStrength :: Proven)"]
    test_query_allowed_actions_2["let conscience = AgentConscience :: new (vec ! [inv])"]
    test_query_allowed_actions_3["let allowed = conscience . query_allowed_actions ('test_fn')"]
    test_query_allowed_actions_4["macro assert"]
    test_query_allowed_actions_5["EXIT"]
    test_query_allowed_actions_0 --> test_query_allowed_actions_1
    test_query_allowed_actions_1 --> test_query_allowed_actions_2
    test_query_allowed_actions_2 --> test_query_allowed_actions_3
    test_query_allowed_actions_3 --> test_query_allowed_actions_4
    test_query_allowed_actions_4 --> test_query_allowed_actions_5
```

