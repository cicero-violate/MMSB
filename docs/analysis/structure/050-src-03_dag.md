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

## File: MMSB/src/03_dag/graph_validator.rs

- Layer(s): 03_dag
- Language coverage: Rust (9)
- Element types: Function (5), Impl (1), Module (1), Struct (2)
- Total elements: 9

### Elements

- [Rust | Struct] `GraphValidationReport` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct GraphValidationReport { pub has_cycle : bool , pub cycle : Vec < PageID > , pub visited...`
- [Rust | Struct] `GraphValidator` (line 0, pub)
  - Signature: `pub struct GraphValidator < 'a > { graph : & 'a ShadowPageGraph , }`
  - Generics: 'a
- [Rust | Function] `detects_cycle` (line 0, priv)
  - Signature: `# [test] fn detects_cycle () { let graph = ShadowPageGraph :: default () ; graph . add_edge (PageID (1) , PageID (2) ...`
  - Calls: ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, add_edge, PageID, PageID, GraphValidator::new, detect_cycles
- [Rust | Impl] `impl < 'a > GraphValidator < 'a > { pub fn new (graph : & 'a ShadowPageGraph) -> Self { Self { graph } } pub fn detect_cycles (& self) -> GraphValidationReport { self . run (None) } pub fn validate_page (& self , root : PageID) -> GraphValidationReport { self . run (Some (root)) } fn run (& self , root : Option < PageID >) -> GraphValidationReport { let start = Instant :: now () ; let adjacency = self . graph . adjacency . read () . clone () ; let nodes : Vec < PageID > = match root { Some (root) => reachable (& adjacency , root) , None => adjacency . keys () . copied () . collect () , } ; let mut index = 0usize ; let mut stack = Vec :: new () ; let mut indices = HashMap :: new () ; let mut lowlink = HashMap :: new () ; let mut on_stack = HashSet :: new () ; let mut cycle = Vec :: new () ; let mut has_cycle = false ; for node in nodes { if has_cycle { break ; } if ! indices . contains_key (& node) { strong_connect (node , & adjacency , & mut index , & mut stack , & mut indices , & mut lowlink , & mut on_stack , & mut cycle , & mut has_cycle ,) ; } } GraphValidationReport { has_cycle , cycle , visited : indices . len () , duration : start . elapsed () , } } } . self_ty` (line 0, priv)
- [Rust | Function] `is_self_loop` (line 0, priv)
  - Signature: `fn is_self_loop (adjacency : & HashMap < PageID , Vec < (PageID , crate :: dag :: edge_types :: EdgeType) > > , node ...`
  - Calls: unwrap_or, map, get, any, iter
- [Rust | Function] `per_page_validation` (line 0, priv)
  - Signature: `# [test] fn per_page_validation () { let graph = ShadowPageGraph :: default () ; graph . add_edge (PageID (1) , PageI...`
  - Calls: ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, GraphValidator::new, validate_page, PageID
- [Rust | Function] `reachable` (line 0, priv)
  - Signature: `fn reachable (adjacency : & HashMap < PageID , Vec < (PageID , crate :: dag :: edge_types :: EdgeType) > > , root : P...`
  - Calls: HashSet::new, pop, insert, get, push, collect, into_iter
- [Rust | Function] `strong_connect` (line 0, priv)
  - Signature: `fn strong_connect (node : PageID , adjacency : & HashMap < PageID , Vec < (PageID , crate :: dag :: edge_types :: Edg...`
  - Calls: insert, insert, push, insert, get, contains_key, strong_connect, unwrap, get, unwrap, get_mut, min, contains, unwrap, get, unwrap, get_mut, min, unwrap, get, unwrap, get, Vec::new, unwrap, pop, remove, push, len, is_self_loop
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/03_dag/mod.rs

- Layer(s): 03_dag
- Language coverage: Rust (6)
- Element types: Module (6)
- Total elements: 6

### Elements

- [Rust | Module] `cycle_detection` (line 0, pub)
- [Rust | Module] `edge_types` (line 0, pub)
- [Rust | Module] `graph_validator` (line 0, pub)
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

