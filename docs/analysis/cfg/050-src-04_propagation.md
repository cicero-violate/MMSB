# CFG Group: src/04_propagation

## Function: `enqueue_sparse`

- File: MMSB/src/04_propagation/sparse_message_passing.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    enqueue_sparse_0["ENTRY"]
    enqueue_sparse_1["queue . push (command)"]
    enqueue_sparse_2["EXIT"]
    enqueue_sparse_0 --> enqueue_sparse_1
    enqueue_sparse_1 --> enqueue_sparse_2
```

## Function: `passthrough`

- File: MMSB/src/04_propagation/propagation_fastpath.rs
- Branches: 0
- Loops: 0
- Nodes: 2
- Edges: 1

```mermaid
flowchart TD
    passthrough_0["ENTRY"]
    passthrough_1["EXIT"]
    passthrough_0 --> passthrough_1
```

