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

