# Structure Group: src/03_dag

## File: MMSB/src/03_dag/DependencyGraph.jl

- Layer(s): 03_dag
- Language coverage: Julia (13)
- Element types: Function (12), Module (1)
- Total elements: 13

### Elements

- [Julia | Module] `DependencyGraph` (line 8, pub)
- [Julia | Function] `add_edge!` (line 33, pub)
  - Signature: `add_edge!(graph::ShadowPageGraph, parent::PageID, child::PageID, edge_type::EdgeType)`
  - Calls: has_cycle_after_add, haskey, lock, push!
- [Julia | Function] `remove_edge!` (line 69, pub)
  - Signature: `remove_edge!(graph::ShadowPageGraph, parent::PageID, child::PageID)`
  - Calls: filter!, haskey, lock
- [Julia | Function] `has_edge` (line 88, pub)
  - Signature: `has_edge(graph::ShadowPageGraph, parent::PageID, child::PageID)::Bool`
  - Calls: any, haskey, lock
- [Julia | Function] `get_children` (line 106, pub)
  - Signature: `get_children(graph::ShadowPageGraph, parent::PageID)::Vector{Tuple{PageID, EdgeType}}`
  - Calls: get, lock
- [Julia | Function] `get_parents` (line 120, pub)
  - Signature: `get_parents(graph::ShadowPageGraph, child::PageID)::Vector{Tuple{PageID, EdgeType}}`
  - Calls: get, lock
- [Julia | Function] `find_descendants` (line 137, pub)
  - Signature: `find_descendants(graph::ShadowPageGraph, root::PageID)::Set{PageID}`
  - Calls: get_children, isempty, lock, popfirst!, push!
- [Julia | Function] `find_ancestors` (line 167, pub)
  - Signature: `find_ancestors(graph::ShadowPageGraph, node::PageID)::Set{PageID}`
  - Calls: get_parents, isempty, lock, popfirst!, push!
- [Julia | Function] `detect_cycles` (line 201, pub)
  - Signature: `detect_cycles(graph::ShadowPageGraph)::Union{Vector{PageID}, Nothing}`
  - Calls: dfs_cycle_detect, keys, lock
- [Julia | Function] `dfs_cycle_detect` (line 236, pub)
  - Signature: `dfs_cycle_detect(graph::ShadowPageGraph, node::PageID, color::Dict{PageID, Symbol}, parent::Dict{PageID, PageID})`
  - Calls: dfs_cycle_detect, get_children, push!, reverse
- [Julia | Function] `topological_order` (line 280, pub)
  - Signature: `topological_order(graph::ShadowPageGraph)::Vector{PageID}`
  - Calls: GraphCycleError, get, get_children, isempty, keys, length, lock, popfirst!, processed, push!, throw
- [Julia | Function] `reverse_postorder` (line 339, pub)
  - Signature: `reverse_postorder(graph::ShadowPageGraph, start::PageID)::Vector{PageID}`
  - Calls: dfs_postorder, get_children, lock, push!
- [Julia | Function] `compute_closure` (line 373, pub)
  - Signature: `compute_closure(graph::ShadowPageGraph, roots::Vector{PageID})::Set{PageID}`
  - Calls: find_descendants, union!

## File: MMSB/src/03_dag/EventSystem.jl

- Layer(s): 03_dag
- Language coverage: Julia (13)
- Element types: Function (11), Module (1), Struct (1)
- Total elements: 13

### Elements

- [Julia | Module] `EventSystem` (line 8, pub)
- [Julia | Struct] `EventSubscription` (line 52, pub)
  - Signature: `struct EventSubscription`
- [Julia | Function] `EventSubscription` (line 57, pub)
  - Signature: `EventSubscription(id::UInt64, event_type::EventType, handler::EventHandler, filter::Union{Function, Nothing}`
- [Julia | Function] `emit_event!` (line 78, pub)
  - Signature: `emit_event!(state::MMSBState, event_type::EventType, data...)`
  - Calls: copy, get, lock
- [Julia | Function] `subscribe!` (line 102, pub)
  - Signature: `subscribe!(event_type::EventType, handler::EventHandler; filter::Union{Function, Nothing}`
- [Julia | Function] `unsubscribe!` (line 118, pub)
  - Signature: `unsubscribe!(sub::EventSubscription)`
  - Calls: filter!, haskey, lock
- [Julia | Function] `log_event!` (line 129, pub)
  - Signature: `log_event!(state::MMSBState, event_type::EventType, data)`
- [Julia | Function] `clear_subscriptions!` (line 138, pub)
  - Signature: `clear_subscriptions!()`
  - Calls: empty!, lock
- [Julia | Function] `get_subscription_count` (line 150, pub)
  - Signature: `get_subscription_count(event_type::EventType)::Int`
  - Calls: get, length, lock
- [Julia | Function] `create_debug_subscriber` (line 161, pub)
  - Signature: `create_debug_subscriber(event_types::Vector{EventType}; verbose::Bool`
- [Julia | Function] `create_logging_subscriber` (line 182, pub)
  - Signature: `create_logging_subscriber(state::MMSBState, log_page_id::PageID)::Vector{EventSubscription}`
  - Calls: instances, log_event_to_page!, subscribe!
- [Julia | Function] `_serialize_event` (line 190, pub)
  - Signature: `_serialize_event(event_type::EventType, data)::Vector{UInt8}`
  - Calls: IOBuffer, collect, serialize, take!
- [Julia | Function] `log_event_to_page!` (line 202, pub)
  - Signature: `log_event_to_page!(state::MMSBState, page_id::PageID, event_type::EventType, data)`
  - Calls: _serialize_event, falses, get_page, length, min, read_page

## File: MMSB/src/03_dag/GraphDSL.jl

- Layer(s): 03_dag
- Language coverage: Julia (2)
- Element types: Function (1), Module (1)
- Total elements: 2

### Elements

- [Julia | Module] `GraphDSL` (line 1, pub)
- [Julia | Function] `node` (line 9, pub)
  - Signature: `node(id)`

## File: MMSB/src/03_dag/ShadowPageGraph.jl

- Layer(s): 03_dag
- Language coverage: Julia (12)
- Element types: Function (10), Module (1), Struct (1)
- Total elements: 12

### Elements

- [Julia | Module] `GraphTypes` (line 8, pub)
- [Julia | Struct] `ShadowPageGraph` (line 31, pub)
  - Signature: `mutable struct ShadowPageGraph`
- [Julia | Function] `ShadowPageGraph` (line 36, pub)
  - Signature: `ShadowPageGraph()`
  - Calls: ReentrantLock, new
- [Julia | Function] `_ensure_vertex!` (line 48, pub)
  - Signature: `_ensure_vertex!(graph::ShadowPageGraph, node::PageID)`
  - Calls: haskey
- [Julia | Function] `add_dependency!` (line 63, pub)
  - Signature: `add_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID, edge_type::EdgeType)`
  - Calls: GraphCycleError, UInt64, _ensure_vertex!, has_cycle, lock, push!, remove_dependency!, throw
- [Julia | Function] `remove_dependency!` (line 86, pub)
  - Signature: `remove_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID)`
  - Calls: filter!, haskey, lock
- [Julia | Function] `get_children` (line 102, pub)
  - Signature: `get_children(graph::ShadowPageGraph, parent::PageID)::Vector{Tuple{PageID, EdgeType}}`
  - Calls: copy, get, lock
- [Julia | Function] `get_parents` (line 113, pub)
  - Signature: `get_parents(graph::ShadowPageGraph, child::PageID)::Vector{Tuple{PageID, EdgeType}}`
  - Calls: copy, get, lock
- [Julia | Function] `_dfs_has_cycle` (line 122, pub)
  - Signature: `_dfs_has_cycle(graph::ShadowPageGraph, node::PageID, visited::Dict{PageID, Symbol})`
  - Calls: _dfs_has_cycle, get
- [Julia | Function] `has_cycle` (line 141, pub)
  - Signature: `has_cycle(graph::ShadowPageGraph)::Bool`
  - Calls: _dfs_has_cycle, keys
- [Julia | Function] `_all_vertices` (line 154, pub)
  - Signature: `_all_vertices(graph::ShadowPageGraph)`
  - Calls: collect, keys, push!
- [Julia | Function] `topological_sort` (line 170, pub)
  - Signature: `topological_sort(graph::ShadowPageGraph)::Vector{PageID}`
  - Calls: GraphCycleError, _all_vertices, get, isempty, length, lock, popfirst!, push!, throw

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

