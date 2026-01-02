# CFG Group: src/050_scc_compressor.rs

## Function: `test_scc_compression_cycle`

- File: src/050_scc_compressor.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    test_scc_compression_cycle_0["ENTRY"]
    test_scc_compression_cycle_1["let mut graph = DiGraph :: new ()"]
    test_scc_compression_cycle_2["let a = graph . add_node ('A' . to_string ())"]
    test_scc_compression_cycle_3["let b = graph . add_node ('B' . to_string ())"]
    test_scc_compression_cycle_4["let c = graph . add_node ('C' . to_string ())"]
    test_scc_compression_cycle_5["graph . add_edge (a , b , ())"]
    test_scc_compression_cycle_6["graph . add_edge (b , c , ())"]
    test_scc_compression_cycle_7["graph . add_edge (c , a , ())"]
    test_scc_compression_cycle_8["let compression = SccCompression :: new (graph)"]
    test_scc_compression_cycle_9["macro assert"]
    test_scc_compression_cycle_10["macro assert_eq"]
    test_scc_compression_cycle_11["macro assert_eq"]
    test_scc_compression_cycle_12["EXIT"]
    test_scc_compression_cycle_0 --> test_scc_compression_cycle_1
    test_scc_compression_cycle_1 --> test_scc_compression_cycle_2
    test_scc_compression_cycle_2 --> test_scc_compression_cycle_3
    test_scc_compression_cycle_3 --> test_scc_compression_cycle_4
    test_scc_compression_cycle_4 --> test_scc_compression_cycle_5
    test_scc_compression_cycle_5 --> test_scc_compression_cycle_6
    test_scc_compression_cycle_6 --> test_scc_compression_cycle_7
    test_scc_compression_cycle_7 --> test_scc_compression_cycle_8
    test_scc_compression_cycle_8 --> test_scc_compression_cycle_9
    test_scc_compression_cycle_9 --> test_scc_compression_cycle_10
    test_scc_compression_cycle_10 --> test_scc_compression_cycle_11
    test_scc_compression_cycle_11 --> test_scc_compression_cycle_12
```

## Function: `test_scc_compression_dag`

- File: src/050_scc_compressor.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    test_scc_compression_dag_0["ENTRY"]
    test_scc_compression_dag_1["let mut graph = DiGraph :: new ()"]
    test_scc_compression_dag_2["let a = graph . add_node ('A' . to_string ())"]
    test_scc_compression_dag_3["let b = graph . add_node ('B' . to_string ())"]
    test_scc_compression_dag_4["let c = graph . add_node ('C' . to_string ())"]
    test_scc_compression_dag_5["graph . add_edge (a , b , ())"]
    test_scc_compression_dag_6["graph . add_edge (b , c , ())"]
    test_scc_compression_dag_7["let compression = SccCompression :: new (graph)"]
    test_scc_compression_dag_8["macro assert"]
    test_scc_compression_dag_9["macro assert_eq"]
    test_scc_compression_dag_10["macro assert_eq"]
    test_scc_compression_dag_11["EXIT"]
    test_scc_compression_dag_0 --> test_scc_compression_dag_1
    test_scc_compression_dag_1 --> test_scc_compression_dag_2
    test_scc_compression_dag_2 --> test_scc_compression_dag_3
    test_scc_compression_dag_3 --> test_scc_compression_dag_4
    test_scc_compression_dag_4 --> test_scc_compression_dag_5
    test_scc_compression_dag_5 --> test_scc_compression_dag_6
    test_scc_compression_dag_6 --> test_scc_compression_dag_7
    test_scc_compression_dag_7 --> test_scc_compression_dag_8
    test_scc_compression_dag_8 --> test_scc_compression_dag_9
    test_scc_compression_dag_9 --> test_scc_compression_dag_10
    test_scc_compression_dag_10 --> test_scc_compression_dag_11
```

## Function: `test_scc_compression_mixed`

- File: src/050_scc_compressor.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    test_scc_compression_mixed_0["ENTRY"]
    test_scc_compression_mixed_1["let mut graph = DiGraph :: new ()"]
    test_scc_compression_mixed_2["let a = graph . add_node ('A' . to_string ())"]
    test_scc_compression_mixed_3["let b = graph . add_node ('B' . to_string ())"]
    test_scc_compression_mixed_4["let c = graph . add_node ('C' . to_string ())"]
    test_scc_compression_mixed_5["let d = graph . add_node ('D' . to_string ())"]
    test_scc_compression_mixed_6["graph . add_edge (a , b , ())"]
    test_scc_compression_mixed_7["graph . add_edge (b , c , ())"]
    test_scc_compression_mixed_8["graph . add_edge (c , b , ())"]
    test_scc_compression_mixed_9["graph . add_edge (a , d , ())"]
    test_scc_compression_mixed_10["let compression = SccCompression :: new (graph)"]
    test_scc_compression_mixed_11["macro assert"]
    test_scc_compression_mixed_12["macro assert_eq"]
    test_scc_compression_mixed_13["macro assert_eq"]
    test_scc_compression_mixed_14["macro assert_eq"]
    test_scc_compression_mixed_15["EXIT"]
    test_scc_compression_mixed_0 --> test_scc_compression_mixed_1
    test_scc_compression_mixed_1 --> test_scc_compression_mixed_2
    test_scc_compression_mixed_2 --> test_scc_compression_mixed_3
    test_scc_compression_mixed_3 --> test_scc_compression_mixed_4
    test_scc_compression_mixed_4 --> test_scc_compression_mixed_5
    test_scc_compression_mixed_5 --> test_scc_compression_mixed_6
    test_scc_compression_mixed_6 --> test_scc_compression_mixed_7
    test_scc_compression_mixed_7 --> test_scc_compression_mixed_8
    test_scc_compression_mixed_8 --> test_scc_compression_mixed_9
    test_scc_compression_mixed_9 --> test_scc_compression_mixed_10
    test_scc_compression_mixed_10 --> test_scc_compression_mixed_11
    test_scc_compression_mixed_11 --> test_scc_compression_mixed_12
    test_scc_compression_mixed_12 --> test_scc_compression_mixed_13
    test_scc_compression_mixed_13 --> test_scc_compression_mixed_14
    test_scc_compression_mixed_14 --> test_scc_compression_mixed_15
```

