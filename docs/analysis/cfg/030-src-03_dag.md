# CFG Group: src/03_dag

## Function: `detects_cycle`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    detects_cycle_0["ENTRY"]
    detects_cycle_1["let graph = ShadowPageGraph :: default ()"]
    detects_cycle_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    detects_cycle_3["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    detects_cycle_4["graph . add_edge (PageID (3) , PageID (1) , EdgeType :: Data)"]
    detects_cycle_5["let validator = GraphValidator :: new (& graph)"]
    detects_cycle_6["let report = validator . detect_cycles ()"]
    detects_cycle_7["macro assert"]
    detects_cycle_8["macro assert"]
    detects_cycle_9["EXIT"]
    detects_cycle_0 --> detects_cycle_1
    detects_cycle_1 --> detects_cycle_2
    detects_cycle_2 --> detects_cycle_3
    detects_cycle_3 --> detects_cycle_4
    detects_cycle_4 --> detects_cycle_5
    detects_cycle_5 --> detects_cycle_6
    detects_cycle_6 --> detects_cycle_7
    detects_cycle_7 --> detects_cycle_8
    detects_cycle_8 --> detects_cycle_9
```

## Function: `dfs`

- File: MMSB/src/03_dag/cycle_detection.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    dfs_0["ENTRY"]
    dfs_1["match states . get (& node) { Some (VisitState :: Visiting) => return true , ..."]
    dfs_2["states . insert (node , VisitState :: Visiting)"]
    dfs_3["if let Some (children) = adjacency . get (& node)"]
    dfs_4["THEN BB"]
    dfs_5["for (child , _) in children { if dfs (* child , adjacency , states) { return ..."]
    dfs_6["EMPTY ELSE"]
    dfs_7["IF JOIN"]
    dfs_8["states . insert (node , VisitState :: Visited)"]
    dfs_9["false"]
    dfs_10["EXIT"]
    dfs_0 --> dfs_1
    dfs_1 --> dfs_2
    dfs_2 --> dfs_3
    dfs_3 --> dfs_4
    dfs_4 --> dfs_5
    dfs_3 --> dfs_6
    dfs_5 --> dfs_7
    dfs_6 --> dfs_7
    dfs_7 --> dfs_8
    dfs_8 --> dfs_9
    dfs_9 --> dfs_10
```

## Function: `has_cycle`

- File: MMSB/src/03_dag/cycle_detection.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    has_cycle_0["ENTRY"]
    has_cycle_1["let adjacency = graph . adjacency . read () . clone ()"]
    has_cycle_2["let mut states : HashMap < PageID , VisitState > = HashMap :: new ()"]
    has_cycle_3["item"]
    has_cycle_4["for node in adjacency . keys () { if dfs (* node , & adjacency , & mut states..."]
    has_cycle_5["false"]
    has_cycle_6["EXIT"]
    has_cycle_0 --> has_cycle_1
    has_cycle_1 --> has_cycle_2
    has_cycle_2 --> has_cycle_3
    has_cycle_3 --> has_cycle_4
    has_cycle_4 --> has_cycle_5
    has_cycle_5 --> has_cycle_6
```

## Function: `is_self_loop`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_self_loop_0["ENTRY"]
    is_self_loop_1["adjacency . get (& node) . map (| edges | edges . iter () . any (| (target , ..."]
    is_self_loop_2["EXIT"]
    is_self_loop_0 --> is_self_loop_1
    is_self_loop_1 --> is_self_loop_2
```

## Function: `per_page_validation`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    per_page_validation_0["ENTRY"]
    per_page_validation_1["let graph = ShadowPageGraph :: default ()"]
    per_page_validation_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    per_page_validation_3["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    per_page_validation_4["let validator = GraphValidator :: new (& graph)"]
    per_page_validation_5["let report = validator . validate_page (PageID (1))"]
    per_page_validation_6["macro assert"]
    per_page_validation_7["macro assert"]
    per_page_validation_8["EXIT"]
    per_page_validation_0 --> per_page_validation_1
    per_page_validation_1 --> per_page_validation_2
    per_page_validation_2 --> per_page_validation_3
    per_page_validation_3 --> per_page_validation_4
    per_page_validation_4 --> per_page_validation_5
    per_page_validation_5 --> per_page_validation_6
    per_page_validation_6 --> per_page_validation_7
    per_page_validation_7 --> per_page_validation_8
```

## Function: `reachable`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    reachable_0["ENTRY"]
    reachable_1["let mut visited = HashSet :: new ()"]
    reachable_2["let mut stack = vec ! [root]"]
    reachable_3["while let Some (node) = stack . pop () { if visited . insert (node) { if let ..."]
    reachable_4["visited . into_iter () . collect ()"]
    reachable_5["EXIT"]
    reachable_0 --> reachable_1
    reachable_1 --> reachable_2
    reachable_2 --> reachable_3
    reachable_3 --> reachable_4
    reachable_4 --> reachable_5
```

## Function: `strong_connect`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 4
- Loops: 1
- Nodes: 36
- Edges: 40

```mermaid
flowchart TD
    strong_connect_0["ENTRY"]
    strong_connect_1["indices . insert (node , * index)"]
    strong_connect_2["lowlink . insert (node , * index)"]
    strong_connect_3["* index += 1"]
    strong_connect_4["stack . push (node)"]
    strong_connect_5["on_stack . insert (node)"]
    strong_connect_6["if let Some (children) = adjacency . get (& node)"]
    strong_connect_7["THEN BB"]
    strong_connect_8["for (child , _) in children { if ! indices . contains_key (child) { strong_co..."]
    strong_connect_9["EMPTY ELSE"]
    strong_connect_10["IF JOIN"]
    strong_connect_11["let node_low = * lowlink . get (& node) . unwrap ()"]
    strong_connect_12["let node_index = * indices . get (& node) . unwrap ()"]
    strong_connect_13["if node_low == node_index"]
    strong_connect_14["THEN BB"]
    strong_connect_15["let mut component = Vec :: new ()"]
    strong_connect_16["LOOP"]
    strong_connect_17["LOOP BB"]
    strong_connect_18["let popped = stack . pop () . unwrap ()"]
    strong_connect_19["on_stack . remove (& popped)"]
    strong_connect_20["component . push (popped)"]
    strong_connect_21["if popped == node"]
    strong_connect_22["THEN BB"]
    strong_connect_23["break"]
    strong_connect_24["EMPTY ELSE"]
    strong_connect_25["IF JOIN"]
    strong_connect_26["AFTER LOOP"]
    strong_connect_27["if component . len () > 1 || is_self_loop (adjacency , node)"]
    strong_connect_28["THEN BB"]
    strong_connect_29["* cycle = component"]
    strong_connect_30["* has_cycle = true"]
    strong_connect_31["EMPTY ELSE"]
    strong_connect_32["IF JOIN"]
    strong_connect_33["EMPTY ELSE"]
    strong_connect_34["IF JOIN"]
    strong_connect_35["EXIT"]
    strong_connect_0 --> strong_connect_1
    strong_connect_1 --> strong_connect_2
    strong_connect_2 --> strong_connect_3
    strong_connect_3 --> strong_connect_4
    strong_connect_4 --> strong_connect_5
    strong_connect_5 --> strong_connect_6
    strong_connect_6 --> strong_connect_7
    strong_connect_7 --> strong_connect_8
    strong_connect_6 --> strong_connect_9
    strong_connect_8 --> strong_connect_10
    strong_connect_9 --> strong_connect_10
    strong_connect_10 --> strong_connect_11
    strong_connect_11 --> strong_connect_12
    strong_connect_12 --> strong_connect_13
    strong_connect_13 --> strong_connect_14
    strong_connect_14 --> strong_connect_15
    strong_connect_15 --> strong_connect_16
    strong_connect_16 --> strong_connect_17
    strong_connect_17 --> strong_connect_18
    strong_connect_18 --> strong_connect_19
    strong_connect_19 --> strong_connect_20
    strong_connect_20 --> strong_connect_21
    strong_connect_21 --> strong_connect_22
    strong_connect_22 --> strong_connect_23
    strong_connect_21 --> strong_connect_24
    strong_connect_23 --> strong_connect_25
    strong_connect_24 --> strong_connect_25
    strong_connect_25 --> strong_connect_16
    strong_connect_25 --> strong_connect_26
    strong_connect_26 --> strong_connect_27
    strong_connect_27 --> strong_connect_28
    strong_connect_28 --> strong_connect_29
    strong_connect_29 --> strong_connect_30
    strong_connect_27 --> strong_connect_31
    strong_connect_30 --> strong_connect_32
    strong_connect_31 --> strong_connect_32
    strong_connect_13 --> strong_connect_33
    strong_connect_32 --> strong_connect_34
    strong_connect_33 --> strong_connect_34
    strong_connect_34 --> strong_connect_35
```

## Function: `topological_sort`

- File: MMSB/src/03_dag/shadow_graph_traversal.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    topological_sort_0["ENTRY"]
    topological_sort_1["let mut result = Vec :: new ()"]
    topological_sort_2["let mut in_degree = HashMap :: new ()"]
    topological_sort_3["let mut adjacency = HashMap :: new ()"]
    topological_sort_4["{ for (node , edges) in graph . adjacency . read () . iter () { adjacency . i..."]
    topological_sort_5["let mut queue : VecDeque < PageID > = in_degree . iter () . filter_map (| (& node , & deg) | if deg == 0 { Some (no..."]
    topological_sort_6["while let Some (node) = queue . pop_front () { result . push (node) ; if let ..."]
    topological_sort_7["result"]
    topological_sort_8["EXIT"]
    topological_sort_0 --> topological_sort_1
    topological_sort_1 --> topological_sort_2
    topological_sort_2 --> topological_sort_3
    topological_sort_3 --> topological_sort_4
    topological_sort_4 --> topological_sort_5
    topological_sort_5 --> topological_sort_6
    topological_sort_6 --> topological_sort_7
    topological_sort_7 --> topological_sort_8
```

