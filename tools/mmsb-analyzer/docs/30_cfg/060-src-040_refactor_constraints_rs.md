# CFG Group: src/040_refactor_constraints.rs

## Function: `check_move_allowed`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    check_move_allowed_0["ENTRY"]
    check_move_allowed_1["for constraint in constraints { if constraint . target () == target && constr..."]
    check_move_allowed_2["Ok (())"]
    check_move_allowed_3["EXIT"]
    check_move_allowed_0 --> check_move_allowed_1
    check_move_allowed_1 --> check_move_allowed_2
    check_move_allowed_2 --> check_move_allowed_3
```

## Function: `from_invariant`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    from_invariant_0["ENTRY"]
    from_invariant_1["match & invariant . kind { InvariantKind :: Structural (s) => match s { Struc..."]
    from_invariant_2["EXIT"]
    from_invariant_0 --> from_invariant_1
    from_invariant_1 --> from_invariant_2
```

## Function: `generate_constraints`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    generate_constraints_0["ENTRY"]
    generate_constraints_1["analysis . invariants . iter () . filter_map (from_invariant) . collect ()"]
    generate_constraints_2["EXIT"]
    generate_constraints_0 --> generate_constraints_1
    generate_constraints_1 --> generate_constraints_2
```

## Function: `test_check_move_allowed`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_check_move_allowed_0["ENTRY"]
    test_check_move_allowed_1["let constraints = vec ! [RefactorConstraint :: NoMove { target : 'test_fn' . to_string () , rea..."]
    test_check_move_allowed_2["let result = check_move_allowed ('test_fn' , 'a.rs' , 'b.rs' , & constraints)"]
    test_check_move_allowed_3["macro assert"]
    test_check_move_allowed_4["macro assert"]
    test_check_move_allowed_5["EXIT"]
    test_check_move_allowed_0 --> test_check_move_allowed_1
    test_check_move_allowed_1 --> test_check_move_allowed_2
    test_check_move_allowed_2 --> test_check_move_allowed_3
    test_check_move_allowed_3 --> test_check_move_allowed_4
    test_check_move_allowed_4 --> test_check_move_allowed_5
```

## Function: `test_check_move_allowed_blocking`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_check_move_allowed_blocking_0["ENTRY"]
    test_check_move_allowed_blocking_1["let constraints = vec ! [RefactorConstraint :: NoMove { target : 'test_fn' . to_string () , rea..."]
    test_check_move_allowed_blocking_2["let result = check_move_allowed ('test_fn' , 'old.rs' , 'new.rs' , & constraints)"]
    test_check_move_allowed_blocking_3["macro assert"]
    test_check_move_allowed_blocking_4["macro assert"]
    test_check_move_allowed_blocking_5["EXIT"]
    test_check_move_allowed_blocking_0 --> test_check_move_allowed_blocking_1
    test_check_move_allowed_blocking_1 --> test_check_move_allowed_blocking_2
    test_check_move_allowed_blocking_2 --> test_check_move_allowed_blocking_3
    test_check_move_allowed_blocking_3 --> test_check_move_allowed_blocking_4
    test_check_move_allowed_blocking_4 --> test_check_move_allowed_blocking_5
```

## Function: `test_check_move_allowed_non_blocking`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_check_move_allowed_non_blocking_0["ENTRY"]
    test_check_move_allowed_non_blocking_1["let constraints = vec ! [RefactorConstraint :: NoMove { target : 'test_fn' . to_string () , rea..."]
    test_check_move_allowed_non_blocking_2["let result = check_move_allowed ('test_fn' , 'old.rs' , 'new.rs' , & constraints)"]
    test_check_move_allowed_non_blocking_3["macro assert"]
    test_check_move_allowed_non_blocking_4["EXIT"]
    test_check_move_allowed_non_blocking_0 --> test_check_move_allowed_non_blocking_1
    test_check_move_allowed_non_blocking_1 --> test_check_move_allowed_non_blocking_2
    test_check_move_allowed_non_blocking_2 --> test_check_move_allowed_non_blocking_3
    test_check_move_allowed_non_blocking_3 --> test_check_move_allowed_non_blocking_4
```

## Function: `test_constraint_is_blocking`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_constraint_is_blocking_0["ENTRY"]
    test_constraint_is_blocking_1["let proven = RefactorConstraint :: NoMove { target : 'fn1' . to_string () , reason : 'test..."]
    test_constraint_is_blocking_2["macro assert"]
    test_constraint_is_blocking_3["let heuristic = RefactorConstraint :: NoMove { target : 'fn2' . to_string () , reason : 'test..."]
    test_constraint_is_blocking_4["macro assert"]
    test_constraint_is_blocking_5["EXIT"]
    test_constraint_is_blocking_0 --> test_constraint_is_blocking_1
    test_constraint_is_blocking_1 --> test_constraint_is_blocking_2
    test_constraint_is_blocking_2 --> test_constraint_is_blocking_3
    test_constraint_is_blocking_3 --> test_constraint_is_blocking_4
    test_constraint_is_blocking_4 --> test_constraint_is_blocking_5
```

## Function: `test_from_invariant_layer_fixed`

- File: src/040_refactor_constraints.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_from_invariant_layer_fixed_0["ENTRY"]
    test_from_invariant_layer_fixed_1["let inv = Invariant :: new ('test_fn' . to_string () , 'test.rs' . to_string () , Invar..."]
    test_from_invariant_layer_fixed_2["let constraint = from_invariant (& inv) . unwrap ()"]
    test_from_invariant_layer_fixed_3["match constraint { RefactorConstraint :: FixedLayer { target , layer , .. } =..."]
    test_from_invariant_layer_fixed_4["EXIT"]
    test_from_invariant_layer_fixed_0 --> test_from_invariant_layer_fixed_1
    test_from_invariant_layer_fixed_1 --> test_from_invariant_layer_fixed_2
    test_from_invariant_layer_fixed_2 --> test_from_invariant_layer_fixed_3
    test_from_invariant_layer_fixed_3 --> test_from_invariant_layer_fixed_4
```

