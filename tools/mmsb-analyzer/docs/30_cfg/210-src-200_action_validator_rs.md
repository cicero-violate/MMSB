# CFG Group: src/200_action_validator.rs

## Function: `check_move_allowed`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    check_move_allowed_0["ENTRY"]
    check_move_allowed_1["let action = AgentAction :: MoveFunction { name : name . to_string () , from : from . clon..."]
    check_move_allowed_2["match validate_action (& action , constraints) { Ok (_) => Ok (()) , Err (vio..."]
    check_move_allowed_3["EXIT"]
    check_move_allowed_0 --> check_move_allowed_1
    check_move_allowed_1 --> check_move_allowed_2
    check_move_allowed_2 --> check_move_allowed_3
```

## Function: `extract_layer`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    extract_layer_0["ENTRY"]
    extract_layer_1["path . file_name () . and_then (| n | n . to_str ()) . and_then (| s | { let ..."]
    extract_layer_2["EXIT"]
    extract_layer_0 --> extract_layer_1
    extract_layer_1 --> extract_layer_2
```

## Function: `test_extract_layer`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_extract_layer_0["ENTRY"]
    test_extract_layer_1["macro assert_eq"]
    test_extract_layer_2["macro assert_eq"]
    test_extract_layer_3["macro assert_eq"]
    test_extract_layer_4["EXIT"]
    test_extract_layer_0 --> test_extract_layer_1
    test_extract_layer_1 --> test_extract_layer_2
    test_extract_layer_2 --> test_extract_layer_3
    test_extract_layer_3 --> test_extract_layer_4
```

## Function: `test_validate_allowed_action`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_validate_allowed_action_0["ENTRY"]
    test_validate_allowed_action_1["let constraints = vec ! [RefactorConstraint :: NoMove { target : 'other_fn' . to_string () , re..."]
    test_validate_allowed_action_2["let action = AgentAction :: MoveFunction { name : 'test_fn' . to_string () , from : PathBu..."]
    test_validate_allowed_action_3["let result = validate_action (& action , & constraints)"]
    test_validate_allowed_action_4["macro assert"]
    test_validate_allowed_action_5["EXIT"]
    test_validate_allowed_action_0 --> test_validate_allowed_action_1
    test_validate_allowed_action_1 --> test_validate_allowed_action_2
    test_validate_allowed_action_2 --> test_validate_allowed_action_3
    test_validate_allowed_action_3 --> test_validate_allowed_action_4
    test_validate_allowed_action_4 --> test_validate_allowed_action_5
```

## Function: `test_validate_layer_fixed_constraint`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_validate_layer_fixed_constraint_0["ENTRY"]
    test_validate_layer_fixed_constraint_1["let constraints = vec ! [RefactorConstraint :: FixedLayer { target : 'test_fn' . to_string () ,..."]
    test_validate_layer_fixed_constraint_2["let action = AgentAction :: MoveFunction { name : 'test_fn' . to_string () , from : PathBu..."]
    test_validate_layer_fixed_constraint_3["let result = validate_action (& action , & constraints)"]
    test_validate_layer_fixed_constraint_4["macro assert"]
    test_validate_layer_fixed_constraint_5["EXIT"]
    test_validate_layer_fixed_constraint_0 --> test_validate_layer_fixed_constraint_1
    test_validate_layer_fixed_constraint_1 --> test_validate_layer_fixed_constraint_2
    test_validate_layer_fixed_constraint_2 --> test_validate_layer_fixed_constraint_3
    test_validate_layer_fixed_constraint_3 --> test_validate_layer_fixed_constraint_4
    test_validate_layer_fixed_constraint_4 --> test_validate_layer_fixed_constraint_5
```

## Function: `test_validate_no_move_constraint`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_validate_no_move_constraint_0["ENTRY"]
    test_validate_no_move_constraint_1["let constraints = vec ! [RefactorConstraint :: NoMove { target : 'test_fn' . to_string () , rea..."]
    test_validate_no_move_constraint_2["let action = AgentAction :: MoveFunction { name : 'test_fn' . to_string () , from : PathBu..."]
    test_validate_no_move_constraint_3["let result = validate_action (& action , & constraints)"]
    test_validate_no_move_constraint_4["macro assert"]
    test_validate_no_move_constraint_5["let violations = result . unwrap_err ()"]
    test_validate_no_move_constraint_6["macro assert_eq"]
    test_validate_no_move_constraint_7["macro assert_eq"]
    test_validate_no_move_constraint_8["macro assert"]
    test_validate_no_move_constraint_9["EXIT"]
    test_validate_no_move_constraint_0 --> test_validate_no_move_constraint_1
    test_validate_no_move_constraint_1 --> test_validate_no_move_constraint_2
    test_validate_no_move_constraint_2 --> test_validate_no_move_constraint_3
    test_validate_no_move_constraint_3 --> test_validate_no_move_constraint_4
    test_validate_no_move_constraint_4 --> test_validate_no_move_constraint_5
    test_validate_no_move_constraint_5 --> test_validate_no_move_constraint_6
    test_validate_no_move_constraint_6 --> test_validate_no_move_constraint_7
    test_validate_no_move_constraint_7 --> test_validate_no_move_constraint_8
    test_validate_no_move_constraint_8 --> test_validate_no_move_constraint_9
```

## Function: `test_validate_preserve_signature`

- File: src/200_action_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_validate_preserve_signature_0["ENTRY"]
    test_validate_preserve_signature_1["let constraints = vec ! [RefactorConstraint :: PreserveSignature { target : 'test_fn' . to_stri..."]
    test_validate_preserve_signature_2["let action = AgentAction :: ChangeSignature { name : 'test_fn' . to_string () , old_sig : ..."]
    test_validate_preserve_signature_3["let result = validate_action (& action , & constraints)"]
    test_validate_preserve_signature_4["macro assert"]
    test_validate_preserve_signature_5["EXIT"]
    test_validate_preserve_signature_0 --> test_validate_preserve_signature_1
    test_validate_preserve_signature_1 --> test_validate_preserve_signature_2
    test_validate_preserve_signature_2 --> test_validate_preserve_signature_3
    test_validate_preserve_signature_3 --> test_validate_preserve_signature_4
    test_validate_preserve_signature_4 --> test_validate_preserve_signature_5
```

## Function: `validate_action`

- File: src/200_action_validator.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    validate_action_0["ENTRY"]
    validate_action_1["let mut violations = Vec :: new ()"]
    validate_action_2["for (idx , constraint) in constraints . iter () . enumerate () { match (actio..."]
    validate_action_3["if violations . is_empty ()"]
    validate_action_4["THEN BB"]
    validate_action_5["Ok (())"]
    validate_action_6["ELSE BB"]
    validate_action_7["{ Err (violations) }"]
    validate_action_8["IF JOIN"]
    validate_action_9["EXIT"]
    validate_action_0 --> validate_action_1
    validate_action_1 --> validate_action_2
    validate_action_2 --> validate_action_3
    validate_action_3 --> validate_action_4
    validate_action_4 --> validate_action_5
    validate_action_3 --> validate_action_6
    validate_action_6 --> validate_action_7
    validate_action_5 --> validate_action_8
    validate_action_7 --> validate_action_8
    validate_action_8 --> validate_action_9
```

