# CFG Group: src/05_adaptive

## Function: `test_locality_cost_empty`

- File: MMSB/src/05_adaptive/memory_layout.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_locality_cost_empty_0["ENTRY"]
    test_locality_cost_empty_1["let layout = MemoryLayout :: new (4096)"]
    test_locality_cost_empty_2["let pattern = AccessPattern { coaccesses : HashMap :: new () , }"]
    test_locality_cost_empty_3["macro assert_eq"]
    test_locality_cost_empty_4["EXIT"]
    test_locality_cost_empty_0 --> test_locality_cost_empty_1
    test_locality_cost_empty_1 --> test_locality_cost_empty_2
    test_locality_cost_empty_2 --> test_locality_cost_empty_3
    test_locality_cost_empty_3 --> test_locality_cost_empty_4
```

## Function: `test_locality_optimizer`

- File: MMSB/src/05_adaptive/locality_optimizer.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_locality_optimizer_0["ENTRY"]
    test_locality_optimizer_1["let mut opt = LocalityOptimizer :: new (4096)"]
    test_locality_optimizer_2["opt . add_edge (1 , 2 , 10.0)"]
    test_locality_optimizer_3["opt . add_edge (2 , 3 , 5.0)"]
    test_locality_optimizer_4["let ordering = opt . compute_ordering ()"]
    test_locality_optimizer_5["macro assert_eq"]
    test_locality_optimizer_6["let addrs = opt . assign_addresses (& ordering)"]
    test_locality_optimizer_7["macro assert_eq"]
    test_locality_optimizer_8["EXIT"]
    test_locality_optimizer_0 --> test_locality_optimizer_1
    test_locality_optimizer_1 --> test_locality_optimizer_2
    test_locality_optimizer_2 --> test_locality_optimizer_3
    test_locality_optimizer_3 --> test_locality_optimizer_4
    test_locality_optimizer_4 --> test_locality_optimizer_5
    test_locality_optimizer_5 --> test_locality_optimizer_6
    test_locality_optimizer_6 --> test_locality_optimizer_7
    test_locality_optimizer_7 --> test_locality_optimizer_8
```

## Function: `test_memory_layout_creation`

- File: MMSB/src/05_adaptive/memory_layout.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_memory_layout_creation_0["ENTRY"]
    test_memory_layout_creation_1["let layout = MemoryLayout :: new (4096)"]
    test_memory_layout_creation_2["macro assert_eq"]
    test_memory_layout_creation_3["macro assert"]
    test_memory_layout_creation_4["EXIT"]
    test_memory_layout_creation_0 --> test_memory_layout_creation_1
    test_memory_layout_creation_1 --> test_memory_layout_creation_2
    test_memory_layout_creation_2 --> test_memory_layout_creation_3
    test_memory_layout_creation_3 --> test_memory_layout_creation_4
```

## Function: `test_optimize_layout`

- File: MMSB/src/05_adaptive/memory_layout.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_optimize_layout_0["ENTRY"]
    test_optimize_layout_1["let mut layout = MemoryLayout :: new (4096)"]
    test_optimize_layout_2["layout . placement . insert (1 , 0)"]
    test_optimize_layout_3["layout . placement . insert (2 , 8192)"]
    test_optimize_layout_4["layout . placement . insert (3 , 16384)"]
    test_optimize_layout_5["let mut pattern = AccessPattern { coaccesses : HashMap :: new () , }"]
    test_optimize_layout_6["pattern . coaccesses . insert ((1 , 2) , 10)"]
    test_optimize_layout_7["pattern . coaccesses . insert ((2 , 3) , 5)"]
    test_optimize_layout_8["layout . optimize_layout (& pattern)"]
    test_optimize_layout_9["macro assert"]
    test_optimize_layout_10["EXIT"]
    test_optimize_layout_0 --> test_optimize_layout_1
    test_optimize_layout_1 --> test_optimize_layout_2
    test_optimize_layout_2 --> test_optimize_layout_3
    test_optimize_layout_3 --> test_optimize_layout_4
    test_optimize_layout_4 --> test_optimize_layout_5
    test_optimize_layout_5 --> test_optimize_layout_6
    test_optimize_layout_6 --> test_optimize_layout_7
    test_optimize_layout_7 --> test_optimize_layout_8
    test_optimize_layout_8 --> test_optimize_layout_9
    test_optimize_layout_9 --> test_optimize_layout_10
```

## Function: `test_page_clustering`

- File: MMSB/src/05_adaptive/page_clustering.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_page_clustering_0["ENTRY"]
    test_page_clustering_1["let mut clusterer = PageClusterer :: new (2)"]
    test_page_clustering_2["let mut coaccesses = HashMap :: new ()"]
    test_page_clustering_3["coaccesses . insert ((1 , 2) , 100)"]
    test_page_clustering_4["coaccesses . insert ((2 , 3) , 50)"]
    test_page_clustering_5["coaccesses . insert ((4 , 5) , 80)"]
    test_page_clustering_6["clusterer . cluster_pages (& coaccesses)"]
    test_page_clustering_7["macro assert"]
    test_page_clustering_8["macro assert"]
    test_page_clustering_9["EXIT"]
    test_page_clustering_0 --> test_page_clustering_1
    test_page_clustering_1 --> test_page_clustering_2
    test_page_clustering_2 --> test_page_clustering_3
    test_page_clustering_3 --> test_page_clustering_4
    test_page_clustering_4 --> test_page_clustering_5
    test_page_clustering_5 --> test_page_clustering_6
    test_page_clustering_6 --> test_page_clustering_7
    test_page_clustering_7 --> test_page_clustering_8
    test_page_clustering_8 --> test_page_clustering_9
```

