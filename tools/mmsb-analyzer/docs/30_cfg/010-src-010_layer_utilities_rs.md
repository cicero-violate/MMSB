# CFG Group: src/010_layer_utilities.rs

## Function: `adjacency_from_edges`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    adjacency_from_edges_0["ENTRY"]
    adjacency_from_edges_1["let mut adjacency : HashMap < String , BTreeSet < String > > = HashMap :: new ()"]
    adjacency_from_edges_2["for ((from , to) , _) in edges_map { adjacency . entry (from . clone ()) . or..."]
    adjacency_from_edges_3["adjacency"]
    adjacency_from_edges_4["EXIT"]
    adjacency_from_edges_0 --> adjacency_from_edges_1
    adjacency_from_edges_1 --> adjacency_from_edges_2
    adjacency_from_edges_2 --> adjacency_from_edges_3
    adjacency_from_edges_3 --> adjacency_from_edges_4
```

## Function: `build_result`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 16
- Edges: 16

```mermaid
flowchart TD
    build_result_0["ENTRY"]
    build_result_1["let adjacency = adjacency_from_edges (& edges_map)"]
    build_result_2["let (mut ordered_layers , cycles) = topo_sort (& nodes , & adjacency)"]
    build_result_3["if let Some (pos) = ordered_layers . iter () . position (| layer | layer == '..."]
    build_result_4["THEN BB"]
    build_result_5["let root_layer = ordered_layers . remove (pos)"]
    build_result_6["ordered_layers . insert (0 , root_layer)"]
    build_result_7["EMPTY ELSE"]
    build_result_8["IF JOIN"]
    build_result_9["let rank = layer_rank_map (& ordered_layers)"]
    build_result_10["let mut ordered_files = files . to_vec ()"]
    build_result_11["ordered_files . sort_by (| a , b | { let mmsb_a = is_mmsb_main (a) ; let mmsb..."]
    build_result_12["let edges = edges_map . into_iter () . map (| ((from , to) , references) | LayerEdge { vi..."]
    build_result_13["let graph = LayerGraph { ordered_layers , edges , cycles , unresolved , }"]
    build_result_14["Ok ((ordered_files , graph))"]
    build_result_15["EXIT"]
    build_result_0 --> build_result_1
    build_result_1 --> build_result_2
    build_result_2 --> build_result_3
    build_result_3 --> build_result_4
    build_result_4 --> build_result_5
    build_result_5 --> build_result_6
    build_result_3 --> build_result_7
    build_result_6 --> build_result_8
    build_result_7 --> build_result_8
    build_result_8 --> build_result_9
    build_result_9 --> build_result_10
    build_result_10 --> build_result_11
    build_result_11 --> build_result_12
    build_result_12 --> build_result_13
    build_result_13 --> build_result_14
    build_result_14 --> build_result_15
```

## Function: `cluster_target_path`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    cluster_target_path_0["ENTRY"]
    cluster_target_path_1["if ! is_core_module_path (& target)"]
    cluster_target_path_2["THEN BB"]
    cluster_target_path_3["return target"]
    cluster_target_path_4["EMPTY ELSE"]
    cluster_target_path_5["IF JOIN"]
    cluster_target_path_6["let prefix = target . file_stem () . and_then (| name | name . to_str ()) . and_then (| st..."]
    cluster_target_path_7["let file_name = format ! ('{:03}_cluster_{:03}.rs' , prefix , idx + 1)"]
    cluster_target_path_8["let dir = members . first () . and_then (| member | member . file . parent ()) . unwrap..."]
    cluster_target_path_9["dir . join (file_name)"]
    cluster_target_path_10["EXIT"]
    cluster_target_path_0 --> cluster_target_path_1
    cluster_target_path_1 --> cluster_target_path_2
    cluster_target_path_2 --> cluster_target_path_3
    cluster_target_path_1 --> cluster_target_path_4
    cluster_target_path_3 --> cluster_target_path_5
    cluster_target_path_4 --> cluster_target_path_5
    cluster_target_path_5 --> cluster_target_path_6
    cluster_target_path_6 --> cluster_target_path_7
    cluster_target_path_7 --> cluster_target_path_8
    cluster_target_path_8 --> cluster_target_path_9
    cluster_target_path_9 --> cluster_target_path_10
```

## Function: `collect_cluster_plans`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    collect_cluster_plans_0["ENTRY"]
    collect_cluster_plans_1["let mut plans = Vec :: new ()"]
    collect_cluster_plans_2["for (idx , cluster) in clusters . iter () . enumerate () { let all_members = ..."]
    collect_cluster_plans_3["plans . sort_by (| a , b | { use std :: cmp :: Ordering ; b . cohesion . part..."]
    collect_cluster_plans_4["plans"]
    collect_cluster_plans_5["EXIT"]
    collect_cluster_plans_0 --> collect_cluster_plans_1
    collect_cluster_plans_1 --> collect_cluster_plans_2
    collect_cluster_plans_2 --> collect_cluster_plans_3
    collect_cluster_plans_3 --> collect_cluster_plans_4
    collect_cluster_plans_4 --> collect_cluster_plans_5
```

## Function: `compare_dir_layers`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    compare_dir_layers_0["ENTRY"]
    compare_dir_layers_1["let a_name = a . file_name () . and_then (| n | n . to_str ()) . unwrap_or ('')"]
    compare_dir_layers_2["let b_name = b . file_name () . and_then (| n | n . to_str ()) . unwrap_or ('')"]
    compare_dir_layers_3["let a_layer = layer_prefix_value (a_name) . unwrap_or (i32 :: MAX)"]
    compare_dir_layers_4["let b_layer = layer_prefix_value (b_name) . unwrap_or (i32 :: MAX)"]
    compare_dir_layers_5["a_layer . cmp (& b_layer) . then_with (| | a_name . cmp (b_name))"]
    compare_dir_layers_6["EXIT"]
    compare_dir_layers_0 --> compare_dir_layers_1
    compare_dir_layers_1 --> compare_dir_layers_2
    compare_dir_layers_2 --> compare_dir_layers_3
    compare_dir_layers_3 --> compare_dir_layers_4
    compare_dir_layers_4 --> compare_dir_layers_5
    compare_dir_layers_5 --> compare_dir_layers_6
```

## Function: `compare_path_components`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    compare_path_components_0["ENTRY"]
    compare_path_components_1["let a_components : Vec < _ > = a . components () . collect ()"]
    compare_path_components_2["let b_components : Vec < _ > = b . components () . collect ()"]
    compare_path_components_3["let min_len = a_components . len () . min (b_components . len ())"]
    compare_path_components_4["for idx in 0 .. min_len { let a_name = a_components [idx] . as_os_str () . to..."]
    compare_path_components_5["a_components . len () . cmp (& b_components . len ())"]
    compare_path_components_6["EXIT"]
    compare_path_components_0 --> compare_path_components_1
    compare_path_components_1 --> compare_path_components_2
    compare_path_components_2 --> compare_path_components_3
    compare_path_components_3 --> compare_path_components_4
    compare_path_components_4 --> compare_path_components_5
    compare_path_components_5 --> compare_path_components_6
```

## Function: `cyclomatic_complexity`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 12
- Edges: 12

```mermaid
flowchart TD
    cyclomatic_complexity_0["ENTRY"]
    cyclomatic_complexity_1["let edges = cfg . edges . len () as isize"]
    cyclomatic_complexity_2["let nodes = cfg . nodes . len () as isize"]
    cyclomatic_complexity_3["let exits = 1isize"]
    cyclomatic_complexity_4["let cc = edges - nodes + 2 * exits"]
    cyclomatic_complexity_5["if cc <= 0"]
    cyclomatic_complexity_6["THEN BB"]
    cyclomatic_complexity_7["1"]
    cyclomatic_complexity_8["ELSE BB"]
    cyclomatic_complexity_9["{ cc as usize }"]
    cyclomatic_complexity_10["IF JOIN"]
    cyclomatic_complexity_11["EXIT"]
    cyclomatic_complexity_0 --> cyclomatic_complexity_1
    cyclomatic_complexity_1 --> cyclomatic_complexity_2
    cyclomatic_complexity_2 --> cyclomatic_complexity_3
    cyclomatic_complexity_3 --> cyclomatic_complexity_4
    cyclomatic_complexity_4 --> cyclomatic_complexity_5
    cyclomatic_complexity_5 --> cyclomatic_complexity_6
    cyclomatic_complexity_6 --> cyclomatic_complexity_7
    cyclomatic_complexity_5 --> cyclomatic_complexity_8
    cyclomatic_complexity_8 --> cyclomatic_complexity_9
    cyclomatic_complexity_7 --> cyclomatic_complexity_10
    cyclomatic_complexity_9 --> cyclomatic_complexity_10
    cyclomatic_complexity_10 --> cyclomatic_complexity_11
```

## Function: `detect_layer_violation`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    detect_layer_violation_0["ENTRY"]
    detect_layer_violation_1["let current_layer = file_layers . get (& func . file_path) . cloned () . unwrap_or_else (| | func..."]
    detect_layer_violation_2["let current_value = layer_prefix_value (& current_layer) ?"]
    detect_layer_violation_3["let mut violation : Option < (i32 , String) > = None"]
    detect_layer_violation_4["for (callee_idx , _) in outgoing { let callee = & functions [* callee_idx] ; ..."]
    detect_layer_violation_5["violation . map (| (_ , target_layer) | (current_layer , target_layer))"]
    detect_layer_violation_6["EXIT"]
    detect_layer_violation_0 --> detect_layer_violation_1
    detect_layer_violation_1 --> detect_layer_violation_2
    detect_layer_violation_2 --> detect_layer_violation_3
    detect_layer_violation_3 --> detect_layer_violation_4
    detect_layer_violation_4 --> detect_layer_violation_5
    detect_layer_violation_5 --> detect_layer_violation_6
```

## Function: `detect_layer_violations`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    detect_layer_violations_0["ENTRY"]
    detect_layer_violations_1["let mut violations = Vec :: new ()"]
    detect_layer_violations_2["for edge in graph . edge_references () { let from = & graph [edge . source ()..."]
    detect_layer_violations_3["violations"]
    detect_layer_violations_4["EXIT"]
    detect_layer_violations_0 --> detect_layer_violations_1
    detect_layer_violations_1 --> detect_layer_violations_2
    detect_layer_violations_2 --> detect_layer_violations_3
    detect_layer_violations_3 --> detect_layer_violations_4
```

## Function: `insert_sorted`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 9
- Edges: 9

```mermaid
flowchart TD
    insert_sorted_0["ENTRY"]
    insert_sorted_1["let mut inserted = false"]
    insert_sorted_2["for idx in 0 .. queue . len () { if value < queue [idx] { queue . insert (idx..."]
    insert_sorted_3["if ! inserted"]
    insert_sorted_4["THEN BB"]
    insert_sorted_5["queue . push_back (value)"]
    insert_sorted_6["EMPTY ELSE"]
    insert_sorted_7["IF JOIN"]
    insert_sorted_8["EXIT"]
    insert_sorted_0 --> insert_sorted_1
    insert_sorted_1 --> insert_sorted_2
    insert_sorted_2 --> insert_sorted_3
    insert_sorted_3 --> insert_sorted_4
    insert_sorted_4 --> insert_sorted_5
    insert_sorted_3 --> insert_sorted_6
    insert_sorted_5 --> insert_sorted_7
    insert_sorted_6 --> insert_sorted_7
    insert_sorted_7 --> insert_sorted_8
```

## Function: `is_core_module_path`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    is_core_module_path_0["ENTRY"]
    is_core_module_path_1["let Some (stem) = path . file_stem () . and_then (| name | name . to_str ()) else { return false ; }"]
    is_core_module_path_2["stem . starts_with ('040_dependency') || stem . starts_with ('060_layer_core')"]
    is_core_module_path_3["EXIT"]
    is_core_module_path_0 --> is_core_module_path_1
    is_core_module_path_1 --> is_core_module_path_2
    is_core_module_path_2 --> is_core_module_path_3
```

## Function: `is_layer_violation`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_layer_violation_0["ENTRY"]
    is_layer_violation_1["match (layer_prefix_value (from) , layer_prefix_value (to)) { (Some (a) , Som..."]
    is_layer_violation_2["EXIT"]
    is_layer_violation_0 --> is_layer_violation_1
    is_layer_violation_1 --> is_layer_violation_2
```

## Function: `is_mmsb_main`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    is_mmsb_main_0["ENTRY"]
    is_mmsb_main_1["path . file_name () . and_then (| n | n . to_str ()) . map (| n | n == 'MMSB...."]
    is_mmsb_main_2["EXIT"]
    is_mmsb_main_0 --> is_mmsb_main_1
    is_mmsb_main_1 --> is_mmsb_main_2
```

## Function: `layer_adheres`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    layer_adheres_0["ENTRY"]
    layer_adheres_1["match (layer_prefix_value (current_layer) , layer_prefix_value (target_layer)..."]
    layer_adheres_2["EXIT"]
    layer_adheres_0 --> layer_adheres_1
    layer_adheres_1 --> layer_adheres_2
```

## Function: `layer_prefix_value`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    layer_prefix_value_0["ENTRY"]
    layer_prefix_value_1["let mut chars = layer . chars ()"]
    layer_prefix_value_2["let mut digits = String :: new ()"]
    layer_prefix_value_3["while let Some (ch) = chars . next () { if ch . is_ascii_digit () { digits . ..."]
    layer_prefix_value_4["if digits . is_empty ()"]
    layer_prefix_value_5["THEN BB"]
    layer_prefix_value_6["None"]
    layer_prefix_value_7["ELSE BB"]
    layer_prefix_value_8["{ digits . parse :: < i32 > () . ok () }"]
    layer_prefix_value_9["IF JOIN"]
    layer_prefix_value_10["EXIT"]
    layer_prefix_value_0 --> layer_prefix_value_1
    layer_prefix_value_1 --> layer_prefix_value_2
    layer_prefix_value_2 --> layer_prefix_value_3
    layer_prefix_value_3 --> layer_prefix_value_4
    layer_prefix_value_4 --> layer_prefix_value_5
    layer_prefix_value_5 --> layer_prefix_value_6
    layer_prefix_value_4 --> layer_prefix_value_7
    layer_prefix_value_7 --> layer_prefix_value_8
    layer_prefix_value_6 --> layer_prefix_value_9
    layer_prefix_value_8 --> layer_prefix_value_9
    layer_prefix_value_9 --> layer_prefix_value_10
```

## Function: `layer_rank_map`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    layer_rank_map_0["ENTRY"]
    layer_rank_map_1["let mut rank = HashMap :: new ()"]
    layer_rank_map_2["for (idx , layer) in order . iter () . enumerate () { rank . insert (layer . ..."]
    layer_rank_map_3["rank"]
    layer_rank_map_4["EXIT"]
    layer_rank_map_0 --> layer_rank_map_1
    layer_rank_map_1 --> layer_rank_map_2
    layer_rank_map_2 --> layer_rank_map_3
    layer_rank_map_3 --> layer_rank_map_4
```

## Function: `node_style`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    node_style_0["ENTRY"]
    node_style_1["match node_type { NodeType :: Entry => ('ellipse' , 'lightgreen' , '\'filled,..."]
    node_style_2["EXIT"]
    node_style_0 --> node_style_1
    node_style_1 --> node_style_2
```

## Function: `parse_cluster_members`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    parse_cluster_members_0["ENTRY"]
    parse_cluster_members_1["cluster . members . iter () . filter_map (| member | { let (file , name) = me..."]
    parse_cluster_members_2["EXIT"]
    parse_cluster_members_0 --> parse_cluster_members_1
    parse_cluster_members_1 --> parse_cluster_members_2
```

## Function: `sort_structural_items`

- File: src/010_layer_utilities.rs
- Branches: 2
- Loops: 0
- Nodes: 28
- Edges: 29

```mermaid
flowchart TD
    sort_structural_items_0["ENTRY"]
    sort_structural_items_1["use"]
    sort_structural_items_2["use"]
    sort_structural_items_3["if items . len () <= 1"]
    sort_structural_items_4["THEN BB"]
    sort_structural_items_5["return"]
    sort_structural_items_6["EMPTY ELSE"]
    sort_structural_items_7["IF JOIN"]
    sort_structural_items_8["let count = items . len ()"]
    sort_structural_items_9["let mut edges : Vec < Vec < usize > > = vec ! [Vec :: new () ; count]"]
    sort_structural_items_10["let mut indegree = vec ! [0usize ; count]"]
    sort_structural_items_11["let mut file_to_items : HashMap < PathBuf , Vec < usize > > = HashMap :: new ()"]
    sort_structural_items_12["for (idx , item) in items . iter () . enumerate () { if let Some (path) = & i..."]
    sort_structural_items_13["for i in 0 .. count { for j in (i + 1) .. count { let req_i = structural_laye..."]
    sort_structural_items_14["for (idx , item) in items . iter () . enumerate () { for file in & item . out..."]
    sort_structural_items_15["let mut ordered_indices = Vec :: with_capacity (count)"]
    sort_structural_items_16["let mut available : Vec < usize > = (0 .. count) . filter (| & i | indegree [i] == 0) . collect ()"]
    sort_structural_items_17["while ! available . is_empty () { available . sort_by (| & a , & b | structur..."]
    sort_structural_items_18["if ordered_indices . len () != count"]
    sort_structural_items_19["THEN BB"]
    sort_structural_items_20["items . sort_by (structural_cmp)"]
    sort_structural_items_21["return"]
    sort_structural_items_22["EMPTY ELSE"]
    sort_structural_items_23["IF JOIN"]
    sort_structural_items_24["let mut reordered = Vec :: with_capacity (count)"]
    sort_structural_items_25["for idx in ordered_indices { reordered . push (items [idx] . clone ()) ; }"]
    sort_structural_items_26["* items = reordered"]
    sort_structural_items_27["EXIT"]
    sort_structural_items_0 --> sort_structural_items_1
    sort_structural_items_1 --> sort_structural_items_2
    sort_structural_items_2 --> sort_structural_items_3
    sort_structural_items_3 --> sort_structural_items_4
    sort_structural_items_4 --> sort_structural_items_5
    sort_structural_items_3 --> sort_structural_items_6
    sort_structural_items_5 --> sort_structural_items_7
    sort_structural_items_6 --> sort_structural_items_7
    sort_structural_items_7 --> sort_structural_items_8
    sort_structural_items_8 --> sort_structural_items_9
    sort_structural_items_9 --> sort_structural_items_10
    sort_structural_items_10 --> sort_structural_items_11
    sort_structural_items_11 --> sort_structural_items_12
    sort_structural_items_12 --> sort_structural_items_13
    sort_structural_items_13 --> sort_structural_items_14
    sort_structural_items_14 --> sort_structural_items_15
    sort_structural_items_15 --> sort_structural_items_16
    sort_structural_items_16 --> sort_structural_items_17
    sort_structural_items_17 --> sort_structural_items_18
    sort_structural_items_18 --> sort_structural_items_19
    sort_structural_items_19 --> sort_structural_items_20
    sort_structural_items_20 --> sort_structural_items_21
    sort_structural_items_18 --> sort_structural_items_22
    sort_structural_items_21 --> sort_structural_items_23
    sort_structural_items_22 --> sort_structural_items_23
    sort_structural_items_23 --> sort_structural_items_24
    sort_structural_items_24 --> sort_structural_items_25
    sort_structural_items_25 --> sort_structural_items_26
    sort_structural_items_26 --> sort_structural_items_27
```

## Function: `structural_cmp`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    structural_cmp_0["ENTRY"]
    structural_cmp_1["let a_required = structural_layer_value (& a . required_layer , i32 :: MAX)"]
    structural_cmp_2["let b_required = structural_layer_value (& b . required_layer , i32 :: MAX)"]
    structural_cmp_3["let a_current = structural_layer_value (& a . current_layer , i32 :: MIN)"]
    structural_cmp_4["let b_current = structural_layer_value (& b . current_layer , i32 :: MIN)"]
    structural_cmp_5["let a_benefit = if a . cost == 0 { 0 } else { (a . benefit . saturating_mul (1000)) / a . cost }"]
    structural_cmp_6["let b_benefit = if b . cost == 0 { 0 } else { (b . benefit . saturating_mul (1000)) / b . cost }"]
    structural_cmp_7["a_required . cmp (& b_required) . then_with (| | b . is_utility . cmp (& a . ..."]
    structural_cmp_8["EXIT"]
    structural_cmp_0 --> structural_cmp_1
    structural_cmp_1 --> structural_cmp_2
    structural_cmp_2 --> structural_cmp_3
    structural_cmp_3 --> structural_cmp_4
    structural_cmp_4 --> structural_cmp_5
    structural_cmp_5 --> structural_cmp_6
    structural_cmp_6 --> structural_cmp_7
    structural_cmp_7 --> structural_cmp_8
```

## Function: `structural_layer_value`

- File: src/010_layer_utilities.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    structural_layer_value_0["ENTRY"]
    structural_layer_value_1["layer . as_ref () . and_then (| value | layer_prefix_value (value)) . unwrap_..."]
    structural_layer_value_2["EXIT"]
    structural_layer_value_0 --> structural_layer_value_1
    structural_layer_value_1 --> structural_layer_value_2
```

## Function: `topo_sort`

- File: src/010_layer_utilities.rs
- Branches: 1
- Loops: 0
- Nodes: 19
- Edges: 19

```mermaid
flowchart TD
    topo_sort_0["ENTRY"]
    topo_sort_1["let mut indegree : HashMap < String , usize > = HashMap :: new ()"]
    topo_sort_2["for node in nodes { indegree . entry (node . clone ()) . or_insert (0) ; }"]
    topo_sort_3["for targets in adjacency . values () { for target in targets { * indegree . e..."]
    topo_sort_4["let mut queue : VecDeque < String > = indegree . iter () . filter_map (| (node , & deg) | if deg == 0 { Some (node ..."]
    topo_sort_5["queue . make_contiguous () . sort ()"]
    topo_sort_6["let mut order = Vec :: new ()"]
    topo_sort_7["while let Some (node) = queue . pop_front () { order . push (node . clone ())..."]
    topo_sort_8["if order . len () != nodes . len ()"]
    topo_sort_9["THEN BB"]
    topo_sort_10["let mut remaining : Vec < _ > = nodes . iter () . filter (| layer | ! order . contains (layer)) . cloned () ...."]
    topo_sort_11["remaining . sort ()"]
    topo_sort_12["let cycles = remaining . clone ()"]
    topo_sort_13["order . extend (remaining)"]
    topo_sort_14["return (order , cycles)"]
    topo_sort_15["EMPTY ELSE"]
    topo_sort_16["IF JOIN"]
    topo_sort_17["(order , Vec :: new ())"]
    topo_sort_18["EXIT"]
    topo_sort_0 --> topo_sort_1
    topo_sort_1 --> topo_sort_2
    topo_sort_2 --> topo_sort_3
    topo_sort_3 --> topo_sort_4
    topo_sort_4 --> topo_sort_5
    topo_sort_5 --> topo_sort_6
    topo_sort_6 --> topo_sort_7
    topo_sort_7 --> topo_sort_8
    topo_sort_8 --> topo_sort_9
    topo_sort_9 --> topo_sort_10
    topo_sort_10 --> topo_sort_11
    topo_sort_11 --> topo_sort_12
    topo_sort_12 --> topo_sort_13
    topo_sort_13 --> topo_sort_14
    topo_sort_8 --> topo_sort_15
    topo_sort_14 --> topo_sort_16
    topo_sort_15 --> topo_sort_16
    topo_sort_16 --> topo_sort_17
    topo_sort_17 --> topo_sort_18
```

