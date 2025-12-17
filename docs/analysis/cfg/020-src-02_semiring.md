# CFG Group: src/02_semiring

## Function: `accumulate`

- File: MMSB/src/02_semiring/semiring_ops.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    accumulate_0["ENTRY"]
    accumulate_1["(semiring . add (left , right) , semiring . mul (left , right))"]
    accumulate_2["EXIT"]
    accumulate_0 --> accumulate_1
    accumulate_1 --> accumulate_2
```

## Function: `detects_impure_function`

- File: MMSB/src/02_semiring/purity_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    detects_impure_function_0["ENTRY"]
    detects_impure_function_1["let validator = PurityValidator :: default ()"]
    detects_impure_function_2["let inputs = vec ! [1u32 , 2 , 3]"]
    detects_impure_function_3["use"]
    detects_impure_function_4["item"]
    detects_impure_function_5["let report = validator . validate_fn (& inputs , | value | { let bump = CALLS . fetch_add ..."]
    detects_impure_function_6["macro assert"]
    detects_impure_function_7["EXIT"]
    detects_impure_function_0 --> detects_impure_function_1
    detects_impure_function_1 --> detects_impure_function_2
    detects_impure_function_2 --> detects_impure_function_3
    detects_impure_function_3 --> detects_impure_function_4
    detects_impure_function_4 --> detects_impure_function_5
    detects_impure_function_5 --> detects_impure_function_6
    detects_impure_function_6 --> detects_impure_function_7
```

## Function: `fold_add`

- File: MMSB/src/02_semiring/semiring_ops.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    fold_add_0["ENTRY"]
    fold_add_1["values . into_iter () . fold (semiring . zero () , | acc , value | semiring ...."]
    fold_add_2["EXIT"]
    fold_add_0 --> fold_add_1
    fold_add_1 --> fold_add_2
```

## Function: `fold_mul`

- File: MMSB/src/02_semiring/semiring_ops.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    fold_mul_0["ENTRY"]
    fold_mul_1["values . into_iter () . fold (semiring . one () , | acc , value | semiring . ..."]
    fold_mul_2["EXIT"]
    fold_mul_0 --> fold_mul_1
    fold_mul_1 --> fold_mul_2
```

## Function: `validates_semiring_operations`

- File: MMSB/src/02_semiring/purity_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    validates_semiring_operations_0["ENTRY"]
    validates_semiring_operations_1["let validator = PurityValidator :: default ()"]
    validates_semiring_operations_2["let bool_samples = vec ! [vec ! [true , false , true] , vec ! [false , false] ,]"]
    validates_semiring_operations_3["macro assert"]
    validates_semiring_operations_4["let tropical_samples = vec ! [vec ! [0.0 , 1.0 , 2.5] , vec ! [3.0 , 4.5] ,]"]
    validates_semiring_operations_5["macro assert"]
    validates_semiring_operations_6["EXIT"]
    validates_semiring_operations_0 --> validates_semiring_operations_1
    validates_semiring_operations_1 --> validates_semiring_operations_2
    validates_semiring_operations_2 --> validates_semiring_operations_3
    validates_semiring_operations_3 --> validates_semiring_operations_4
    validates_semiring_operations_4 --> validates_semiring_operations_5
    validates_semiring_operations_5 --> validates_semiring_operations_6
```

