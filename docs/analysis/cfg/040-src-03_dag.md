# CFG Group: src/03_dag

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

