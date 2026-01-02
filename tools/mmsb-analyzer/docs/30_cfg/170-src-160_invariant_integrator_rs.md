# CFG Group: src/160_invariant_integrator.rs

## Function: `make_simple_analysis`

- File: src/160_invariant_integrator.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    make_simple_analysis_0["ENTRY"]
    make_simple_analysis_1["let mut result = AnalysisResult :: new ()"]
    make_simple_analysis_2["result . add_element (CodeElement { name : 'test_fn' . to_string () , file_pa..."]
    make_simple_analysis_3["let mut call_graph = HashMap :: new ()"]
    make_simple_analysis_4["call_graph . insert ('test_fn' . to_string () , CallGraphNode { function_name..."]
    make_simple_analysis_5["result . call_graph = call_graph"]
    make_simple_analysis_6["result"]
    make_simple_analysis_7["EXIT"]
    make_simple_analysis_0 --> make_simple_analysis_1
    make_simple_analysis_1 --> make_simple_analysis_2
    make_simple_analysis_2 --> make_simple_analysis_3
    make_simple_analysis_3 --> make_simple_analysis_4
    make_simple_analysis_4 --> make_simple_analysis_5
    make_simple_analysis_5 --> make_simple_analysis_6
    make_simple_analysis_6 --> make_simple_analysis_7
```

## Function: `test_detect_all`

- File: src/160_invariant_integrator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_detect_all_0["ENTRY"]
    test_detect_all_1["let analysis = make_simple_analysis ()"]
    test_detect_all_2["let detector = InvariantDetector :: new (& analysis , & analysis . call_graph)"]
    test_detect_all_3["let result = detector . detect_all ()"]
    test_detect_all_4["macro assert"]
    test_detect_all_5["EXIT"]
    test_detect_all_0 --> test_detect_all_1
    test_detect_all_1 --> test_detect_all_2
    test_detect_all_2 --> test_detect_all_3
    test_detect_all_3 --> test_detect_all_4
    test_detect_all_4 --> test_detect_all_5
```

## Function: `test_invariant_detector_creation`

- File: src/160_invariant_integrator.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_invariant_detector_creation_0["ENTRY"]
    test_invariant_detector_creation_1["let analysis = make_simple_analysis ()"]
    test_invariant_detector_creation_2["let detector = InvariantDetector :: new (& analysis , & analysis . call_graph)"]
    test_invariant_detector_creation_3["macro assert_eq"]
    test_invariant_detector_creation_4["EXIT"]
    test_invariant_detector_creation_0 --> test_invariant_detector_creation_1
    test_invariant_detector_creation_1 --> test_invariant_detector_creation_2
    test_invariant_detector_creation_2 --> test_invariant_detector_creation_3
    test_invariant_detector_creation_3 --> test_invariant_detector_creation_4
```

