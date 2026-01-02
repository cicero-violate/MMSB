# CFG Group: src/190_conscience_graph.rs

## Function: `generate_conscience_map`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 26
- Edges: 25

```mermaid
flowchart TD
    generate_conscience_map_0["ENTRY"]
    generate_conscience_map_1["let mut content = String :: new ()"]
    generate_conscience_map_2["content . push_str ('# Conscience Map\n\n')"]
    generate_conscience_map_3["content . push_str ('## Overview\n\n')"]
    generate_conscience_map_4["content . push_str ('This map shows which functions are protected by mechanic..."]
    generate_conscience_map_5["content . push_str ('Functions with blocking invariants cannot be refactored ..."]
    generate_conscience_map_6["let mut by_function : HashMap < String , Vec < & Invariant > > = HashMap :: new ()"]
    generate_conscience_map_7["for inv in invariants { by_function . entry (inv . target . clone ()) . or_de..."]
    generate_conscience_map_8["let total_functions = by_function . len ()"]
    generate_conscience_map_9["let protected_functions = by_function . values () . filter (| invs | invs . iter () . any (| i | i . is..."]
    generate_conscience_map_10["content . push_str (& format ! ('**Total Functions**: {}\n\n' , total_functio..."]
    generate_conscience_map_11["content . push_str (& format ! ('**Protected Functions**: {} ({:.1}%)\n\n' , ..."]
    generate_conscience_map_12["let mut funcs : Vec < _ > = by_function . into_iter () . collect ()"]
    generate_conscience_map_13["funcs . sort_by_key (| (_ , invs) | { - (invs . iter () . filter (| i | i . i..."]
    generate_conscience_map_14["content . push_str ('---\n\n')"]
    generate_conscience_map_15["content . push_str ('## Functions by Protection Level\n\n')"]
    generate_conscience_map_16["for (func , invs) in funcs { let blocking_count = invs . iter () . filter (| ..."]
    generate_conscience_map_17["content . push_str ('---\n\n')"]
    generate_conscience_map_18["content . push_str ('## Legend\n\n')"]
    generate_conscience_map_19["content . push_str ('- ✓ **PROVEN**: Mathematical certainty from graph topo..."]
    generate_conscience_map_20["content . push_str ('- ◆ **EMPIRICAL**: Observed across multiple paths (hig..."]
    generate_conscience_map_21["content . push_str ('- ? **HEURISTIC**: Name-based guess (LOW CONFIDENCE - ve..."]
    generate_conscience_map_22["content . push_str ('- 🔒 **Blocking**: Constraint mechanically enforced\n\n')"]
    generate_conscience_map_23["std :: fs :: write (output_path , content) ?"]
    generate_conscience_map_24["Ok (())"]
    generate_conscience_map_25["EXIT"]
    generate_conscience_map_0 --> generate_conscience_map_1
    generate_conscience_map_1 --> generate_conscience_map_2
    generate_conscience_map_2 --> generate_conscience_map_3
    generate_conscience_map_3 --> generate_conscience_map_4
    generate_conscience_map_4 --> generate_conscience_map_5
    generate_conscience_map_5 --> generate_conscience_map_6
    generate_conscience_map_6 --> generate_conscience_map_7
    generate_conscience_map_7 --> generate_conscience_map_8
    generate_conscience_map_8 --> generate_conscience_map_9
    generate_conscience_map_9 --> generate_conscience_map_10
    generate_conscience_map_10 --> generate_conscience_map_11
    generate_conscience_map_11 --> generate_conscience_map_12
    generate_conscience_map_12 --> generate_conscience_map_13
    generate_conscience_map_13 --> generate_conscience_map_14
    generate_conscience_map_14 --> generate_conscience_map_15
    generate_conscience_map_15 --> generate_conscience_map_16
    generate_conscience_map_16 --> generate_conscience_map_17
    generate_conscience_map_17 --> generate_conscience_map_18
    generate_conscience_map_18 --> generate_conscience_map_19
    generate_conscience_map_19 --> generate_conscience_map_20
    generate_conscience_map_20 --> generate_conscience_map_21
    generate_conscience_map_21 --> generate_conscience_map_22
    generate_conscience_map_22 --> generate_conscience_map_23
    generate_conscience_map_23 --> generate_conscience_map_24
    generate_conscience_map_24 --> generate_conscience_map_25
```

## Function: `generate_conscience_stats`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    generate_conscience_stats_0["ENTRY"]
    generate_conscience_stats_1["let mut by_function : HashMap < String , Vec < & Invariant > > = HashMap :: new ()"]
    generate_conscience_stats_2["for inv in invariants { by_function . entry (inv . target . clone ()) . or_de..."]
    generate_conscience_stats_3["let total_functions = by_function . len ()"]
    generate_conscience_stats_4["let protected_functions = by_function . values () . filter (| invs | invs . iter () . any (| i | i . is..."]
    generate_conscience_stats_5["let proven_count = invariants . iter () . filter (| i | matches ! (i . strength , InvariantStren..."]
    generate_conscience_stats_6["let empirical_count = invariants . iter () . filter (| i | matches ! (i . strength , InvariantStren..."]
    generate_conscience_stats_7["let heuristic_count = invariants . iter () . filter (| i | matches ! (i . strength , InvariantStren..."]
    generate_conscience_stats_8["ConscienceStats { total_functions , protected_functions , total_invariants : ..."]
    generate_conscience_stats_9["EXIT"]
    generate_conscience_stats_0 --> generate_conscience_stats_1
    generate_conscience_stats_1 --> generate_conscience_stats_2
    generate_conscience_stats_2 --> generate_conscience_stats_3
    generate_conscience_stats_3 --> generate_conscience_stats_4
    generate_conscience_stats_4 --> generate_conscience_stats_5
    generate_conscience_stats_5 --> generate_conscience_stats_6
    generate_conscience_stats_6 --> generate_conscience_stats_7
    generate_conscience_stats_7 --> generate_conscience_stats_8
    generate_conscience_stats_8 --> generate_conscience_stats_9
```

## Function: `kind_name`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    kind_name_0["ENTRY"]
    kind_name_1["match & inv . kind { InvariantKind :: Structural (s) => match s { StructuralI..."]
    kind_name_2["EXIT"]
    kind_name_0 --> kind_name_1
    kind_name_1 --> kind_name_2
```

## Function: `make_test_invariant`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    make_test_invariant_0["ENTRY"]
    make_test_invariant_1["Invariant :: new (name . to_string () , 'test.rs' . to_string () , kind , str..."]
    make_test_invariant_2["EXIT"]
    make_test_invariant_0 --> make_test_invariant_1
    make_test_invariant_1 --> make_test_invariant_2
```

## Function: `strength_emoji`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    strength_emoji_0["ENTRY"]
    strength_emoji_1["match inv . strength { InvariantStrength :: Proven => '✓' , InvariantStreng..."]
    strength_emoji_2["EXIT"]
    strength_emoji_0 --> strength_emoji_1
    strength_emoji_1 --> strength_emoji_2
```

## Function: `test_generate_stats`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_generate_stats_0["ENTRY"]
    test_generate_stats_1["let invariants = vec ! [make_test_invariant ('fn1' , InvariantKind :: Structural (StructuralIn..."]
    test_generate_stats_2["let stats = generate_conscience_stats (& invariants)"]
    test_generate_stats_3["macro assert_eq"]
    test_generate_stats_4["macro assert_eq"]
    test_generate_stats_5["macro assert_eq"]
    test_generate_stats_6["macro assert_eq"]
    test_generate_stats_7["EXIT"]
    test_generate_stats_0 --> test_generate_stats_1
    test_generate_stats_1 --> test_generate_stats_2
    test_generate_stats_2 --> test_generate_stats_3
    test_generate_stats_3 --> test_generate_stats_4
    test_generate_stats_4 --> test_generate_stats_5
    test_generate_stats_5 --> test_generate_stats_6
    test_generate_stats_6 --> test_generate_stats_7
```

## Function: `test_strength_emoji`

- File: src/190_conscience_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_strength_emoji_0["ENTRY"]
    test_strength_emoji_1["let proven = make_test_invariant ('test' , InvariantKind :: Structural (StructuralInvarian..."]
    test_strength_emoji_2["let empirical = make_test_invariant ('test' , InvariantKind :: Semantic (SemanticInvariant ::..."]
    test_strength_emoji_3["let heuristic = make_test_invariant ('test' , InvariantKind :: Semantic (SemanticInvariant ::..."]
    test_strength_emoji_4["macro assert_eq"]
    test_strength_emoji_5["macro assert_eq"]
    test_strength_emoji_6["macro assert_eq"]
    test_strength_emoji_7["EXIT"]
    test_strength_emoji_0 --> test_strength_emoji_1
    test_strength_emoji_1 --> test_strength_emoji_2
    test_strength_emoji_2 --> test_strength_emoji_3
    test_strength_emoji_3 --> test_strength_emoji_4
    test_strength_emoji_4 --> test_strength_emoji_5
    test_strength_emoji_5 --> test_strength_emoji_6
    test_strength_emoji_6 --> test_strength_emoji_7
```

