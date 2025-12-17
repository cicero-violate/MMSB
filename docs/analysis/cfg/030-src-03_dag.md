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
- Branches: 5
- Loops: 1
- Nodes: 19
- Edges: 23

```mermaid
flowchart TD
    dfs_0["ENTRY"]
    dfs_1["match states . get (& node)"]
    dfs_2["arm Some (VisitState :: Visiting)"]
    dfs_3["return true"]
    dfs_4["arm Some (VisitState :: Visited)"]
    dfs_5["return false"]
    dfs_6["arm None"]
    dfs_7["match join"]
    dfs_8["states . insert (node , VisitState :: Visiting)"]
    dfs_9["if let Some (children) = adjacency . get (& node)"]
    dfs_10["for (child , _) in children"]
    dfs_11["if dfs (* child , adjacency , states)"]
    dfs_12["return true"]
    dfs_13["if join"]
    dfs_14["after for"]
    dfs_15["if join"]
    dfs_16["states . insert (node , VisitState :: Visited)"]
    dfs_17["false"]
    dfs_18["EXIT"]
    dfs_0 --> dfs_1
    dfs_1 --> dfs_2
    dfs_2 --> dfs_3
    dfs_1 --> dfs_4
    dfs_4 --> dfs_5
    dfs_1 --> dfs_6
    dfs_3 --> dfs_7
    dfs_5 --> dfs_7
    dfs_6 --> dfs_7
    dfs_7 --> dfs_8
    dfs_8 --> dfs_9
    dfs_9 --> dfs_10
    dfs_10 --> dfs_11
    dfs_11 --> dfs_12
    dfs_12 --> dfs_13
    dfs_11 --> dfs_13
    dfs_13 --> dfs_10
    dfs_10 --> dfs_14
    dfs_14 --> dfs_15
    dfs_9 --> dfs_15
    dfs_15 --> dfs_16
    dfs_16 --> dfs_17
    dfs_17 --> dfs_18
```

## Function: `has_cycle`

- File: MMSB/src/03_dag/cycle_detection.rs
- Branches: 1
- Loops: 1
- Nodes: 11
- Edges: 12

```mermaid
flowchart TD
    has_cycle_0["ENTRY"]
    has_cycle_1["let adjacency = graph . adjacency . read () . clone ()"]
    has_cycle_2["let mut states : HashMap < PageID , VisitState > = HashMap :: new ()"]
    has_cycle_3["item"]
    has_cycle_4["for node in adjacency . keys ()"]
    has_cycle_5["if dfs (* node , & adjacency , & mut states)"]
    has_cycle_6["return true"]
    has_cycle_7["if join"]
    has_cycle_8["after for"]
    has_cycle_9["false"]
    has_cycle_10["EXIT"]
    has_cycle_0 --> has_cycle_1
    has_cycle_1 --> has_cycle_2
    has_cycle_2 --> has_cycle_3
    has_cycle_3 --> has_cycle_4
    has_cycle_4 --> has_cycle_5
    has_cycle_5 --> has_cycle_6
    has_cycle_6 --> has_cycle_7
    has_cycle_5 --> has_cycle_7
    has_cycle_7 --> has_cycle_4
    has_cycle_4 --> has_cycle_8
    has_cycle_8 --> has_cycle_9
    has_cycle_9 --> has_cycle_10
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
- Branches: 2
- Loops: 2
- Nodes: 14
- Edges: 17

```mermaid
flowchart TD
    reachable_0["ENTRY"]
    reachable_1["let mut visited = HashSet :: new ()"]
    reachable_2["let mut stack = vec ! [root]"]
    reachable_3["while let Some (node) = stack . pop ()"]
    reachable_4["if visited . insert (node)"]
    reachable_5["if let Some (children) = adjacency . get (& node)"]
    reachable_6["for (child , _) in children"]
    reachable_7["stack . push (* child)"]
    reachable_8["after for"]
    reachable_9["if join"]
    reachable_10["if join"]
    reachable_11["after while"]
    reachable_12["visited . into_iter () . collect ()"]
    reachable_13["EXIT"]
    reachable_0 --> reachable_1
    reachable_1 --> reachable_2
    reachable_2 --> reachable_3
    reachable_3 --> reachable_4
    reachable_4 --> reachable_5
    reachable_5 --> reachable_6
    reachable_6 --> reachable_7
    reachable_7 --> reachable_6
    reachable_6 --> reachable_8
    reachable_8 --> reachable_9
    reachable_5 --> reachable_9
    reachable_9 --> reachable_10
    reachable_4 --> reachable_10
    reachable_10 --> reachable_3
    reachable_3 --> reachable_11
    reachable_11 --> reachable_12
    reachable_12 --> reachable_13
```

## Function: `strong_connect`

- File: MMSB/src/03_dag/graph_validator.rs
- Branches: 7
- Loops: 2
- Nodes: 42
- Edges: 50

```mermaid
flowchart TD
    strong_connect_0["ENTRY"]
    strong_connect_1["indices . insert (node , * index)"]
    strong_connect_2["lowlink . insert (node , * index)"]
    strong_connect_3["* index += 1"]
    strong_connect_4["stack . push (node)"]
    strong_connect_5["on_stack . insert (node)"]
    strong_connect_6["if let Some (children) = adjacency . get (& node)"]
    strong_connect_7["for (child , _) in children"]
    strong_connect_8["if ! indices . contains_key (child)"]
    strong_connect_9["strong_connect (* child , adjacency , index , stack , indices , lowlink , on_..."]
    strong_connect_10["let child_low = * lowlink . get (child) . unwrap ()"]
    strong_connect_11["let node_low = lowlink . get_mut (& node) . unwrap ()"]
    strong_connect_12["* node_low = (* node_low) . min (child_low)"]
    strong_connect_13["if on_stack . contains (child)"]
    strong_connect_14["let child_index = * indices . get (child) . unwrap ()"]
    strong_connect_15["let node_low = lowlink . get_mut (& node) . unwrap ()"]
    strong_connect_16["* node_low = (* node_low) . min (child_index)"]
    strong_connect_17["if join"]
    strong_connect_18["if join"]
    strong_connect_19["if * has_cycle"]
    strong_connect_20["return"]
    strong_connect_21["if join"]
    strong_connect_22["after for"]
    strong_connect_23["if join"]
    strong_connect_24["let node_low = * lowlink . get (& node) . unwrap ()"]
    strong_connect_25["let node_index = * indices . get (& node) . unwrap ()"]
    strong_connect_26["if node_low == node_index"]
    strong_connect_27["let mut component = Vec :: new ()"]
    strong_connect_28["loop"]
    strong_connect_29["let popped = stack . pop () . unwrap ()"]
    strong_connect_30["on_stack . remove (& popped)"]
    strong_connect_31["component . push (popped)"]
    strong_connect_32["if popped == node"]
    strong_connect_33["break"]
    strong_connect_34["if join"]
    strong_connect_35["loop break"]
    strong_connect_36["if component . len () > 1 || is_self_loop (adjacency , node)"]
    strong_connect_37["* cycle = component"]
    strong_connect_38["* has_cycle = true"]
    strong_connect_39["if join"]
    strong_connect_40["if join"]
    strong_connect_41["EXIT"]
    strong_connect_0 --> strong_connect_1
    strong_connect_1 --> strong_connect_2
    strong_connect_2 --> strong_connect_3
    strong_connect_3 --> strong_connect_4
    strong_connect_4 --> strong_connect_5
    strong_connect_5 --> strong_connect_6
    strong_connect_6 --> strong_connect_7
    strong_connect_7 --> strong_connect_8
    strong_connect_8 --> strong_connect_9
    strong_connect_9 --> strong_connect_10
    strong_connect_10 --> strong_connect_11
    strong_connect_11 --> strong_connect_12
    strong_connect_8 --> strong_connect_13
    strong_connect_13 --> strong_connect_14
    strong_connect_14 --> strong_connect_15
    strong_connect_15 --> strong_connect_16
    strong_connect_16 --> strong_connect_17
    strong_connect_13 --> strong_connect_17
    strong_connect_12 --> strong_connect_18
    strong_connect_17 --> strong_connect_18
    strong_connect_18 --> strong_connect_19
    strong_connect_19 --> strong_connect_20
    strong_connect_20 --> strong_connect_21
    strong_connect_19 --> strong_connect_21
    strong_connect_21 --> strong_connect_7
    strong_connect_7 --> strong_connect_22
    strong_connect_22 --> strong_connect_23
    strong_connect_6 --> strong_connect_23
    strong_connect_23 --> strong_connect_24
    strong_connect_24 --> strong_connect_25
    strong_connect_25 --> strong_connect_26
    strong_connect_26 --> strong_connect_27
    strong_connect_27 --> strong_connect_28
    strong_connect_28 --> strong_connect_29
    strong_connect_29 --> strong_connect_30
    strong_connect_30 --> strong_connect_31
    strong_connect_31 --> strong_connect_32
    strong_connect_32 --> strong_connect_33
    strong_connect_33 --> strong_connect_34
    strong_connect_32 --> strong_connect_34
    strong_connect_34 --> strong_connect_28
    strong_connect_28 --> strong_connect_35
    strong_connect_35 --> strong_connect_36
    strong_connect_36 --> strong_connect_37
    strong_connect_37 --> strong_connect_38
    strong_connect_38 --> strong_connect_39
    strong_connect_36 --> strong_connect_39
    strong_connect_39 --> strong_connect_40
    strong_connect_26 --> strong_connect_40
    strong_connect_40 --> strong_connect_41
```

## Function: `topological_sort`

- File: MMSB/src/03_dag/shadow_graph_traversal.rs
- Branches: 3
- Loops: 4
- Nodes: 27
- Edges: 33

```mermaid
flowchart TD
    topological_sort_0["ENTRY"]
    topological_sort_1["let mut result = Vec :: new ()"]
    topological_sort_2["let mut in_degree = HashMap :: new ()"]
    topological_sort_3["let mut adjacency = HashMap :: new ()"]
    topological_sort_4["for (node , edges) in graph . adjacency . read () . iter ()"]
    topological_sort_5["adjacency . insert (* node , edges . clone ())"]
    topological_sort_6["in_degree . entry (* node) . or_insert (0)"]
    topological_sort_7["for (child , _) in edges . iter ()"]
    topological_sort_8["* in_degree . entry (* child) . or_insert (0) += 1"]
    topological_sort_9["after for"]
    topological_sort_10["after for"]
    topological_sort_11["let mut queue : VecDeque < PageID > = in_degree . iter () . filter_map (| (& node , & deg) | if deg == 0 { Some (no..."]
    topological_sort_12["while let Some (node) = queue . pop_front ()"]
    topological_sort_13["result . push (node)"]
    topological_sort_14["if let Some (children) = adjacency . get (& node)"]
    topological_sort_15["for (child , _) in children"]
    topological_sort_16["if let Some (deg) = in_degree . get_mut (child)"]
    topological_sort_17["* deg -= 1"]
    topological_sort_18["if * deg == 0"]
    topological_sort_19["queue . push_back (* child)"]
    topological_sort_20["if join"]
    topological_sort_21["if join"]
    topological_sort_22["after for"]
    topological_sort_23["if join"]
    topological_sort_24["after while"]
    topological_sort_25["result"]
    topological_sort_26["EXIT"]
    topological_sort_0 --> topological_sort_1
    topological_sort_1 --> topological_sort_2
    topological_sort_2 --> topological_sort_3
    topological_sort_3 --> topological_sort_4
    topological_sort_4 --> topological_sort_5
    topological_sort_5 --> topological_sort_6
    topological_sort_6 --> topological_sort_7
    topological_sort_7 --> topological_sort_8
    topological_sort_8 --> topological_sort_7
    topological_sort_7 --> topological_sort_9
    topological_sort_9 --> topological_sort_4
    topological_sort_4 --> topological_sort_10
    topological_sort_10 --> topological_sort_11
    topological_sort_11 --> topological_sort_12
    topological_sort_12 --> topological_sort_13
    topological_sort_13 --> topological_sort_14
    topological_sort_14 --> topological_sort_15
    topological_sort_15 --> topological_sort_16
    topological_sort_16 --> topological_sort_17
    topological_sort_17 --> topological_sort_18
    topological_sort_18 --> topological_sort_19
    topological_sort_19 --> topological_sort_20
    topological_sort_18 --> topological_sort_20
    topological_sort_20 --> topological_sort_21
    topological_sort_16 --> topological_sort_21
    topological_sort_21 --> topological_sort_15
    topological_sort_15 --> topological_sort_22
    topological_sort_22 --> topological_sort_23
    topological_sort_14 --> topological_sort_23
    topological_sort_23 --> topological_sort_12
    topological_sort_12 --> topological_sort_24
    topological_sort_24 --> topological_sort_25
    topological_sort_25 --> topological_sort_26
```

