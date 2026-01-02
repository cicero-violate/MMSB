# CFG Group: src/410_dead_code_test_boundaries.rs

## Function: `find_test_callers`

- File: src/410_dead_code_test_boundaries.rs
- Branches: 1
- Loops: 0
- Nodes: 13
- Edges: 13

```mermaid
flowchart TD
    find_test_callers_0["ENTRY"]
    find_test_callers_1["if test_symbols . is_empty ()"]
    find_test_callers_2["THEN BB"]
    find_test_callers_3["return Vec :: new ()"]
    find_test_callers_4["EMPTY ELSE"]
    find_test_callers_5["IF JOIN"]
    find_test_callers_6["let reverse = build_reverse_call_graph (call_graph)"]
    find_test_callers_7["let mut callers = Vec :: new ()"]
    find_test_callers_8["let mut visited = HashSet :: new ()"]
    find_test_callers_9["let mut queue : VecDeque < String > = reverse . get (symbol) . cloned () . unwrap_or_default () . into_iter () . co..."]
    find_test_callers_10["while let Some (caller) = queue . pop_front () { if ! visited . insert (calle..."]
    find_test_callers_11["callers"]
    find_test_callers_12["EXIT"]
    find_test_callers_0 --> find_test_callers_1
    find_test_callers_1 --> find_test_callers_2
    find_test_callers_2 --> find_test_callers_3
    find_test_callers_1 --> find_test_callers_4
    find_test_callers_3 --> find_test_callers_5
    find_test_callers_4 --> find_test_callers_5
    find_test_callers_5 --> find_test_callers_6
    find_test_callers_6 --> find_test_callers_7
    find_test_callers_7 --> find_test_callers_8
    find_test_callers_8 --> find_test_callers_9
    find_test_callers_9 --> find_test_callers_10
    find_test_callers_10 --> find_test_callers_11
    find_test_callers_11 --> find_test_callers_12
```

## Function: `has_test_attr`

- File: src/410_dead_code_test_boundaries.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    has_test_attr_0["ENTRY"]
    has_test_attr_1["attrs . iter () . any (| attr | { let path = attr . path () ; if path . is_id..."]
    has_test_attr_2["EXIT"]
    has_test_attr_0 --> has_test_attr_1
    has_test_attr_1 --> has_test_attr_2
```

## Function: `item_attrs`

- File: src/410_dead_code_test_boundaries.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    item_attrs_0["ENTRY"]
    item_attrs_1["match item { Item :: Fn (item_fn) => & item_fn . attrs , Item :: Struct (item..."]
    item_attrs_2["EXIT"]
    item_attrs_0 --> item_attrs_1
    item_attrs_1 --> item_attrs_2
```

