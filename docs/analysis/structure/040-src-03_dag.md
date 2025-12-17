# Structure Group: src/03_dag

## File: MMSB/src/03_dag/cycle_detection.rs

- Layer(s): 03_dag
- Language coverage: Rust (3)
- Element types: Enum (1), Function (2)
- Total elements: 3

### Elements

- [Rust | Enum] `VisitState` (line 0, priv)
- [Rust | Function] `dfs` (line 0, priv)
  - Signature: `fn dfs (node : PageID , adjacency : & HashMap < PageID , Vec < (PageID , crate :: dag :: edge_types :: EdgeType) > > ...`
  - Calls: get, insert, get, dfs, insert
- [Rust | Function] `has_cycle` (line 0, pub)
  - Signature: `# [doc = " Detect whether the given graph contains a cycle using DFS."] pub fn has_cycle (graph : & ShadowPageGraph) ...`
  - Calls: clone, read, HashMap::new, get, insert, get, dfs, insert, keys, dfs

## File: MMSB/src/03_dag/edge_types.rs

- Layer(s): 03_dag
- Language coverage: Rust (1)
- Element types: Enum (1)
- Total elements: 1

### Elements

- [Rust | Enum] `EdgeType` (line 0, pub)

## File: MMSB/src/03_dag/mod.rs

- Layer(s): 03_dag
- Language coverage: Rust (5)
- Element types: Module (5)
- Total elements: 5

### Elements

- [Rust | Module] `cycle_detection` (line 0, pub)
- [Rust | Module] `edge_types` (line 0, pub)
- [Rust | Module] `shadow_graph` (line 0, pub)
- [Rust | Module] `shadow_graph_mod` (line 0, pub)
- [Rust | Module] `shadow_graph_traversal` (line 0, pub)

## File: MMSB/src/03_dag/shadow_graph.rs

- Layer(s): 03_dag
- Language coverage: Rust (3)
- Element types: Impl (1), Struct (2)
- Total elements: 3

### Elements

- [Rust | Struct] `Edge` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , PartialEq , Eq , Hash)] pub struct Edge { pub from : PageID , pub to : PageID , pub edge_t...`
- [Rust | Struct] `ShadowPageGraph` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct ShadowPageGraph { pub (crate) adjacency : RwLock < HashMap < PageID , Vec < (...`
- [Rust | Impl] `impl ShadowPageGraph { pub fn add_edge (& self , from : PageID , to : PageID , edge_type : EdgeType) { let mut guard = self . adjacency . write () ; guard . entry (from) . or_default () . push ((to , edge_type)) ; } pub fn remove_edge (& self , from : PageID , to : PageID) { if let Some (edges) = self . adjacency . write () . get_mut (& from) { edges . retain (| (target , _) | target != & to) ; } } pub fn descendants (& self , root : PageID) -> HashSet < PageID > { let graph = self . adjacency . read () ; let mut seen = HashSet :: new () ; let mut stack = vec ! [root] ; while let Some (node) = stack . pop () { if seen . insert (node) { if let Some (children) = graph . get (& node) { for (child , _) in children { stack . push (* child) ; } } } } seen } } . self_ty` (line 0, priv)

## File: MMSB/src/03_dag/shadow_graph_traversal.rs

- Layer(s): 03_dag
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `topological_sort` (line 0, pub)
  - Signature: `pub fn topological_sort (graph : & ShadowPageGraph) -> Vec < PageID > { let mut result = Vec :: new () ; let mut in_d...`
  - Calls: Vec::new, HashMap::new, HashMap::new, iter, read, insert, clone, or_insert, entry, iter, or_insert, entry, collect, filter_map, iter, Some, pop_front, push, get, get_mut, push_back

