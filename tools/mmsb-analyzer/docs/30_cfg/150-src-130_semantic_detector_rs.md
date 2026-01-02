# CFG Group: src/130_semantic_detector.rs

## Function: `make_function`

- File: src/130_semantic_detector.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    make_function_0["ENTRY"]
    make_function_1["CodeElement { name : name . to_string () , file_path : 'test.rs' . to_string ..."]
    make_function_2["EXIT"]
    make_function_0 --> make_function_1
    make_function_1 --> make_function_2
```

## Function: `test_detect_idempotent_heuristic`

- File: src/130_semantic_detector.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_detect_idempotent_heuristic_0["ENTRY"]
    test_detect_idempotent_heuristic_1["let elements = vec ! [make_function ('set_value' , 'fn set_value(x: i32)' , Vec :: new ()) ,..."]
    test_detect_idempotent_heuristic_2["let detector = SemanticDetector :: new (& elements)"]
    test_detect_idempotent_heuristic_3["let invariants = detector . detect_idempotent ()"]
    test_detect_idempotent_heuristic_4["macro assert_eq"]
    test_detect_idempotent_heuristic_5["for inv in & invariants { assert ! (matches ! (inv . strength , InvariantStre..."]
    test_detect_idempotent_heuristic_6["EXIT"]
    test_detect_idempotent_heuristic_0 --> test_detect_idempotent_heuristic_1
    test_detect_idempotent_heuristic_1 --> test_detect_idempotent_heuristic_2
    test_detect_idempotent_heuristic_2 --> test_detect_idempotent_heuristic_3
    test_detect_idempotent_heuristic_3 --> test_detect_idempotent_heuristic_4
    test_detect_idempotent_heuristic_4 --> test_detect_idempotent_heuristic_5
    test_detect_idempotent_heuristic_5 --> test_detect_idempotent_heuristic_6
```

## Function: `test_detect_pure_function_heuristic`

- File: src/130_semantic_detector.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_detect_pure_function_heuristic_0["ENTRY"]
    test_detect_pure_function_heuristic_1["let elements = vec ! [make_function ('is_valid' , 'fn is_valid(x: i32) -> bool' , Vec :: new..."]
    test_detect_pure_function_heuristic_2["let detector = SemanticDetector :: new (& elements)"]
    test_detect_pure_function_heuristic_3["let invariants = detector . detect_pure_function ()"]
    test_detect_pure_function_heuristic_4["macro assert"]
    test_detect_pure_function_heuristic_5["for inv in & invariants { assert ! (matches ! (inv . strength , InvariantStre..."]
    test_detect_pure_function_heuristic_6["EXIT"]
    test_detect_pure_function_heuristic_0 --> test_detect_pure_function_heuristic_1
    test_detect_pure_function_heuristic_1 --> test_detect_pure_function_heuristic_2
    test_detect_pure_function_heuristic_2 --> test_detect_pure_function_heuristic_3
    test_detect_pure_function_heuristic_3 --> test_detect_pure_function_heuristic_4
    test_detect_pure_function_heuristic_4 --> test_detect_pure_function_heuristic_5
    test_detect_pure_function_heuristic_5 --> test_detect_pure_function_heuristic_6
```

## Function: `test_detect_type_stable`

- File: src/130_semantic_detector.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_detect_type_stable_0["ENTRY"]
    test_detect_type_stable_1["let elements = vec ! [make_function ('test_fn' , 'fn test_fn(x: i32) -> i32' , Vec :: new ()..."]
    test_detect_type_stable_2["let detector = SemanticDetector :: new (& elements)"]
    test_detect_type_stable_3["let invariants = detector . detect_type_stable ()"]
    test_detect_type_stable_4["macro assert_eq"]
    test_detect_type_stable_5["macro assert_eq"]
    test_detect_type_stable_6["macro assert"]
    test_detect_type_stable_7["EXIT"]
    test_detect_type_stable_0 --> test_detect_type_stable_1
    test_detect_type_stable_1 --> test_detect_type_stable_2
    test_detect_type_stable_2 --> test_detect_type_stable_3
    test_detect_type_stable_3 --> test_detect_type_stable_4
    test_detect_type_stable_4 --> test_detect_type_stable_5
    test_detect_type_stable_5 --> test_detect_type_stable_6
    test_detect_type_stable_6 --> test_detect_type_stable_7
```

## Function: `test_no_pure_for_mutable`

- File: src/130_semantic_detector.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_no_pure_for_mutable_0["ENTRY"]
    test_no_pure_for_mutable_1["let elements = vec ! [make_function ('is_valid' , 'fn is_valid(x: &mut i32) -> bool' , Vec :..."]
    test_no_pure_for_mutable_2["let detector = SemanticDetector :: new (& elements)"]
    test_no_pure_for_mutable_3["let invariants = detector . detect_pure_function ()"]
    test_no_pure_for_mutable_4["let pure_count = invariants . iter () . filter (| inv | inv . target == 'is_valid') . count ()"]
    test_no_pure_for_mutable_5["macro assert_eq"]
    test_no_pure_for_mutable_6["EXIT"]
    test_no_pure_for_mutable_0 --> test_no_pure_for_mutable_1
    test_no_pure_for_mutable_1 --> test_no_pure_for_mutable_2
    test_no_pure_for_mutable_2 --> test_no_pure_for_mutable_3
    test_no_pure_for_mutable_3 --> test_no_pure_for_mutable_4
    test_no_pure_for_mutable_4 --> test_no_pure_for_mutable_5
    test_no_pure_for_mutable_5 --> test_no_pure_for_mutable_6
```

