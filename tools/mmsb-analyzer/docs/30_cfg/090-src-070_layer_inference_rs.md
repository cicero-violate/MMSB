# CFG Group: src/070_layer_inference.rs

## Function: `detect_layer_violations`

- File: src/070_layer_inference.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    detect_layer_violations_0["ENTRY"]
    detect_layer_violations_1["let mut violations = Vec :: new ()"]
    detect_layer_violations_2["for edge in graph . edge_references () { let caller = & graph [edge . source ..."]
    detect_layer_violations_3["violations"]
    detect_layer_violations_4["EXIT"]
    detect_layer_violations_0 --> detect_layer_violations_1
    detect_layer_violations_1 --> detect_layer_violations_2
    detect_layer_violations_2 --> detect_layer_violations_3
    detect_layer_violations_3 --> detect_layer_violations_4
```

## Function: `infer_layers`

- File: src/070_layer_inference.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    infer_layers_0["ENTRY"]
    infer_layers_1["let mut layers : HashMap < NodeIndex , usize > = HashMap :: new ()"]
    infer_layers_2["let mut result : HashMap < String , LayerInfo > = HashMap :: new ()"]
    infer_layers_3["for node_idx in graph . node_indices () { let out_degree = graph . neighbors_..."]
    infer_layers_4["let mut changed = true"]
    infer_layers_5["let mut iteration = 0"]
    infer_layers_6["while changed && iteration < max_iterations { changed = false ; iteration += ..."]
    infer_layers_7["let max_layer = layers . values () . max () . copied () . unwrap_or (0)"]
    infer_layers_8["for node_idx in graph . node_indices () { if ! layers . contains_key (& node_..."]
    infer_layers_9["for (node_idx , & layer) in & layers { let name = graph [* node_idx] . clone ..."]
    infer_layers_10["result"]
    infer_layers_11["EXIT"]
    infer_layers_0 --> infer_layers_1
    infer_layers_1 --> infer_layers_2
    infer_layers_2 --> infer_layers_3
    infer_layers_3 --> infer_layers_4
    infer_layers_4 --> infer_layers_5
    infer_layers_5 --> infer_layers_6
    infer_layers_6 --> infer_layers_7
    infer_layers_7 --> infer_layers_8
    infer_layers_8 --> infer_layers_9
    infer_layers_9 --> infer_layers_10
    infer_layers_10 --> infer_layers_11
```

## Function: `test_detect_layer_violations_none`

- File: src/070_layer_inference.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_detect_layer_violations_none_0["ENTRY"]
    test_detect_layer_violations_none_1["let mut graph = DiGraph :: new ()"]
    test_detect_layer_violations_none_2["let a = graph . add_node ('A' . to_string ())"]
    test_detect_layer_violations_none_3["let b = graph . add_node ('B' . to_string ())"]
    test_detect_layer_violations_none_4["let c = graph . add_node ('C' . to_string ())"]
    test_detect_layer_violations_none_5["graph . add_edge (a , b , ())"]
    test_detect_layer_violations_none_6["graph . add_edge (b , c , ())"]
    test_detect_layer_violations_none_7["let layers = infer_layers (& graph , 100)"]
    test_detect_layer_violations_none_8["let violations = detect_layer_violations (& layers , & graph)"]
    test_detect_layer_violations_none_9["macro assert_eq"]
    test_detect_layer_violations_none_10["EXIT"]
    test_detect_layer_violations_none_0 --> test_detect_layer_violations_none_1
    test_detect_layer_violations_none_1 --> test_detect_layer_violations_none_2
    test_detect_layer_violations_none_2 --> test_detect_layer_violations_none_3
    test_detect_layer_violations_none_3 --> test_detect_layer_violations_none_4
    test_detect_layer_violations_none_4 --> test_detect_layer_violations_none_5
    test_detect_layer_violations_none_5 --> test_detect_layer_violations_none_6
    test_detect_layer_violations_none_6 --> test_detect_layer_violations_none_7
    test_detect_layer_violations_none_7 --> test_detect_layer_violations_none_8
    test_detect_layer_violations_none_8 --> test_detect_layer_violations_none_9
    test_detect_layer_violations_none_9 --> test_detect_layer_violations_none_10
```

## Function: `test_layer_inference_diamond`

- File: src/070_layer_inference.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    test_layer_inference_diamond_0["ENTRY"]
    test_layer_inference_diamond_1["let mut graph = DiGraph :: new ()"]
    test_layer_inference_diamond_2["let a = graph . add_node ('A' . to_string ())"]
    test_layer_inference_diamond_3["let b = graph . add_node ('B' . to_string ())"]
    test_layer_inference_diamond_4["let c = graph . add_node ('C' . to_string ())"]
    test_layer_inference_diamond_5["let d = graph . add_node ('D' . to_string ())"]
    test_layer_inference_diamond_6["graph . add_edge (a , b , ())"]
    test_layer_inference_diamond_7["graph . add_edge (a , c , ())"]
    test_layer_inference_diamond_8["graph . add_edge (b , d , ())"]
    test_layer_inference_diamond_9["graph . add_edge (c , d , ())"]
    test_layer_inference_diamond_10["let layers = infer_layers (& graph , 100)"]
    test_layer_inference_diamond_11["macro assert_eq"]
    test_layer_inference_diamond_12["macro assert_eq"]
    test_layer_inference_diamond_13["macro assert_eq"]
    test_layer_inference_diamond_14["macro assert_eq"]
    test_layer_inference_diamond_15["EXIT"]
    test_layer_inference_diamond_0 --> test_layer_inference_diamond_1
    test_layer_inference_diamond_1 --> test_layer_inference_diamond_2
    test_layer_inference_diamond_2 --> test_layer_inference_diamond_3
    test_layer_inference_diamond_3 --> test_layer_inference_diamond_4
    test_layer_inference_diamond_4 --> test_layer_inference_diamond_5
    test_layer_inference_diamond_5 --> test_layer_inference_diamond_6
    test_layer_inference_diamond_6 --> test_layer_inference_diamond_7
    test_layer_inference_diamond_7 --> test_layer_inference_diamond_8
    test_layer_inference_diamond_8 --> test_layer_inference_diamond_9
    test_layer_inference_diamond_9 --> test_layer_inference_diamond_10
    test_layer_inference_diamond_10 --> test_layer_inference_diamond_11
    test_layer_inference_diamond_11 --> test_layer_inference_diamond_12
    test_layer_inference_diamond_12 --> test_layer_inference_diamond_13
    test_layer_inference_diamond_13 --> test_layer_inference_diamond_14
    test_layer_inference_diamond_14 --> test_layer_inference_diamond_15
```

## Function: `test_layer_inference_simple_dag`

- File: src/070_layer_inference.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    test_layer_inference_simple_dag_0["ENTRY"]
    test_layer_inference_simple_dag_1["let mut graph = DiGraph :: new ()"]
    test_layer_inference_simple_dag_2["let a = graph . add_node ('A' . to_string ())"]
    test_layer_inference_simple_dag_3["let b = graph . add_node ('B' . to_string ())"]
    test_layer_inference_simple_dag_4["let c = graph . add_node ('C' . to_string ())"]
    test_layer_inference_simple_dag_5["graph . add_edge (a , b , ())"]
    test_layer_inference_simple_dag_6["graph . add_edge (b , c , ())"]
    test_layer_inference_simple_dag_7["let layers = infer_layers (& graph , 100)"]
    test_layer_inference_simple_dag_8["macro assert_eq"]
    test_layer_inference_simple_dag_9["macro assert_eq"]
    test_layer_inference_simple_dag_10["macro assert_eq"]
    test_layer_inference_simple_dag_11["EXIT"]
    test_layer_inference_simple_dag_0 --> test_layer_inference_simple_dag_1
    test_layer_inference_simple_dag_1 --> test_layer_inference_simple_dag_2
    test_layer_inference_simple_dag_2 --> test_layer_inference_simple_dag_3
    test_layer_inference_simple_dag_3 --> test_layer_inference_simple_dag_4
    test_layer_inference_simple_dag_4 --> test_layer_inference_simple_dag_5
    test_layer_inference_simple_dag_5 --> test_layer_inference_simple_dag_6
    test_layer_inference_simple_dag_6 --> test_layer_inference_simple_dag_7
    test_layer_inference_simple_dag_7 --> test_layer_inference_simple_dag_8
    test_layer_inference_simple_dag_8 --> test_layer_inference_simple_dag_9
    test_layer_inference_simple_dag_9 --> test_layer_inference_simple_dag_10
    test_layer_inference_simple_dag_10 --> test_layer_inference_simple_dag_11
```

