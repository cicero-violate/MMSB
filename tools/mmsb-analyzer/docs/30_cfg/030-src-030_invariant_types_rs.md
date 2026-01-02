# CFG Group: src/030_invariant_types.rs

## Function: `test_confidence_from_strength`

- File: src/030_invariant_types.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_confidence_from_strength_0["ENTRY"]
    test_confidence_from_strength_1["macro assert_eq"]
    test_confidence_from_strength_2["macro assert_eq"]
    test_confidence_from_strength_3["macro assert_eq"]
    test_confidence_from_strength_4["EXIT"]
    test_confidence_from_strength_0 --> test_confidence_from_strength_1
    test_confidence_from_strength_1 --> test_confidence_from_strength_2
    test_confidence_from_strength_2 --> test_confidence_from_strength_3
    test_confidence_from_strength_3 --> test_confidence_from_strength_4
```

## Function: `test_is_blocking`

- File: src/030_invariant_types.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_is_blocking_0["ENTRY"]
    test_is_blocking_1["let proven_inv = Invariant :: new ('test_fn' . to_string () , 'test.rs' . to_string () , Invar..."]
    test_is_blocking_2["macro assert"]
    test_is_blocking_3["let heuristic_inv = Invariant :: new ('test_fn' . to_string () , 'test.rs' . to_string () , Invar..."]
    test_is_blocking_4["macro assert"]
    test_is_blocking_5["EXIT"]
    test_is_blocking_0 --> test_is_blocking_1
    test_is_blocking_1 --> test_is_blocking_2
    test_is_blocking_2 --> test_is_blocking_3
    test_is_blocking_3 --> test_is_blocking_4
    test_is_blocking_4 --> test_is_blocking_5
```

## Function: `test_stats_calculation`

- File: src/030_invariant_types.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_stats_calculation_0["ENTRY"]
    test_stats_calculation_1["let mut result = InvariantAnalysisResult :: new ()"]
    test_stats_calculation_2["result . add_invariant (Invariant :: new ('fn1' . to_string () , 'test.rs' . ..."]
    test_stats_calculation_3["result . add_invariant (Invariant :: new ('fn2' . to_string () , 'test.rs' . ..."]
    test_stats_calculation_4["result . stats . update_totals ()"]
    test_stats_calculation_5["macro assert_eq"]
    test_stats_calculation_6["macro assert_eq"]
    test_stats_calculation_7["macro assert_eq"]
    test_stats_calculation_8["macro assert_eq"]
    test_stats_calculation_9["EXIT"]
    test_stats_calculation_0 --> test_stats_calculation_1
    test_stats_calculation_1 --> test_stats_calculation_2
    test_stats_calculation_2 --> test_stats_calculation_3
    test_stats_calculation_3 --> test_stats_calculation_4
    test_stats_calculation_4 --> test_stats_calculation_5
    test_stats_calculation_5 --> test_stats_calculation_6
    test_stats_calculation_6 --> test_stats_calculation_7
    test_stats_calculation_7 --> test_stats_calculation_8
    test_stats_calculation_8 --> test_stats_calculation_9
```

