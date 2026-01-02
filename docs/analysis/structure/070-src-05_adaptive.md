# Structure Group: src/05_adaptive

## File: MMSB/src/05_adaptive/locality_optimizer.rs

- Layer(s): 05_adaptive
- Language coverage: Rust (5)
- Element types: Function (1), Impl (1), Module (1), Struct (2)
- Total elements: 5

### Elements

- [Rust | Struct] `LocalityOptimizer` (line 0, pub)
  - Signature: `# [doc = " Locality optimizer for cache-aware page placement"] pub struct LocalityOptimizer { # [doc = " Page depende...`
- [Rust | Struct] `PageEdge` (line 0, pub)
  - Signature: `# [doc = " Graph edge representing page dependency"] # [derive (Debug , Clone)] pub struct PageEdge { pub source : Pa...`
- [Rust | Impl] `impl LocalityOptimizer { pub fn new (page_size : usize) -> Self { Self { edges : Vec :: new () , page_size , } } pub fn add_edge (& mut self , source : PageId , target : PageId , weight : f64) { self . edges . push (PageEdge { source , target , weight }) ; } # [doc = " Compute optimal page ordering using modified topological sort"] # [doc = " that respects locality (BFS-like traversal with weight prioritization)"] pub fn compute_ordering (& self) -> Vec < PageId > { if self . edges . is_empty () { return Vec :: new () ; } let mut adj : HashMap < PageId , Vec < (PageId , f64) > > = HashMap :: new () ; let mut all_pages : Vec < PageId > = Vec :: new () ; for edge in & self . edges { adj . entry (edge . source) . or_default () . push ((edge . target , edge . weight)) ; if ! all_pages . contains (& edge . source) { all_pages . push (edge . source) ; } if ! all_pages . contains (& edge . target) { all_pages . push (edge . target) ; } } let mut ordered = Vec :: new () ; let mut visited = std :: collections :: HashSet :: new () ; all_pages . sort_by (| a , b | { let weight_a : f64 = adj . get (a) . map (| v | v . iter () . map (| (_ , w) | w) . sum ()) . unwrap_or (0.0) ; let weight_b : f64 = adj . get (b) . map (| v | v . iter () . map (| (_ , w) | w) . sum ()) . unwrap_or (0.0) ; weight_b . partial_cmp (& weight_a) . unwrap_or (std :: cmp :: Ordering :: Equal) }) ; for & root in & all_pages { if visited . contains (& root) { continue ; } Self :: dfs_visit (root , & adj , & mut visited , & mut ordered) ; } ordered } fn dfs_visit (node : PageId , adj : & HashMap < PageId , Vec < (PageId , f64) > > , visited : & mut std :: collections :: HashSet < PageId > , ordered : & mut Vec < PageId > ,) { if visited . contains (& node) { return ; } visited . insert (node) ; if let Some (neighbors) = adj . get (& node) { let mut sorted_neighbors = neighbors . clone () ; sorted_neighbors . sort_by (| a , b | b . 1 . partial_cmp (& a . 1) . unwrap_or (std :: cmp :: Ordering :: Equal)) ; for (neighbor , _) in sorted_neighbors { Self :: dfs_visit (neighbor , adj , visited , ordered) ; } } ordered . push (node) ; } # [doc = " Assign physical addresses based on ordering"] pub fn assign_addresses (& self , ordering : & [PageId]) -> HashMap < PageId , PhysAddr > { let mut placement = HashMap :: new () ; for (i , & page_id) in ordering . iter () . enumerate () { placement . insert (page_id , (i as u64) * (self . page_size as u64)) ; } placement } } . self_ty` (line 0, priv)
- [Rust | Function] `test_locality_optimizer` (line 0, priv)
  - Signature: `# [test] fn test_locality_optimizer () { let mut opt = LocalityOptimizer :: new (4096) ; opt . add_edge (1 , 2 , 10.0...`
  - Calls: LocalityOptimizer::new, add_edge, add_edge, compute_ordering, assign_addresses
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/05_adaptive/memory_layout.rs

- Layer(s): 05_adaptive
- Language coverage: Rust (7)
- Element types: Function (3), Impl (1), Module (1), Struct (2)
- Total elements: 7

### Elements

- [Rust | Struct] `AccessPattern` (line 0, pub)
  - Signature: `# [doc = " Access pattern for locality optimization"] # [derive (Debug , Clone)] pub struct AccessPattern { # [doc = ...`
- [Rust | Struct] `MemoryLayout` (line 0, pub)
  - Signature: `# [doc = " Memory layout representing page-to-address mapping"] # [derive (Debug , Clone)] pub struct MemoryLayout { ...`
- [Rust | Impl] `impl MemoryLayout { # [doc = " Create new memory layout"] pub fn new (page_size : usize) -> Self { Self { placement : HashMap :: new () , page_size , } } # [doc = " Compute locality cost: sum of distances weighted by co-access frequency"] pub fn locality_cost (& self , pattern : & AccessPattern) -> f64 { let mut cost = 0.0 ; for ((p1 , p2) , freq) in & pattern . coaccesses { if let (Some (& addr1) , Some (& addr2)) = (self . placement . get (p1) , self . placement . get (p2)) { let distance = if addr1 > addr2 { addr1 - addr2 } else { addr2 - addr1 } ; cost += (distance / self . page_size as u64) as f64 * (* freq as f64) ; } } cost } # [doc = " Reorder pages to minimize locality cost using greedy clustering"] pub fn optimize_layout (& mut self , pattern : & AccessPattern) { let mut pages : Vec < PageId > = self . placement . keys () . copied () . collect () ; if pages . is_empty () { return ; } pages . sort_by_key (| p | { let freq : u64 = pattern . coaccesses . iter () . filter (| ((p1 , p2) , _) | p1 == p || p2 == p) . map (| (_ , f) | f) . sum () ; std :: cmp :: Reverse (freq) }) ; let base_addr = 0u64 ; for (i , page_id) in pages . iter () . enumerate () { let new_addr = base_addr + (i as u64) * (self . page_size as u64) ; self . placement . insert (* page_id , new_addr) ; } } } . self_ty` (line 0, priv)
- [Rust | Function] `test_locality_cost_empty` (line 0, priv)
  - Signature: `# [test] fn test_locality_cost_empty () { let layout = MemoryLayout :: new (4096) ; let pattern = AccessPattern { coa...`
  - Calls: MemoryLayout::new, HashMap::new
- [Rust | Function] `test_memory_layout_creation` (line 0, priv)
  - Signature: `# [test] fn test_memory_layout_creation () { let layout = MemoryLayout :: new (4096) ; assert_eq ! (layout . page_siz...`
  - Calls: MemoryLayout::new
- [Rust | Function] `test_optimize_layout` (line 0, priv)
  - Signature: `# [test] fn test_optimize_layout () { let mut layout = MemoryLayout :: new (4096) ; layout . placement . insert (1 , ...`
  - Calls: MemoryLayout::new, insert, insert, insert, HashMap::new, insert, insert, optimize_layout
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/05_adaptive/mod.rs

- Layer(s): 05_adaptive
- Language coverage: Rust (3)
- Element types: Module (3)
- Total elements: 3

### Elements

- [Rust | Module] `locality_optimizer` (line 0, pub)
- [Rust | Module] `memory_layout` (line 0, pub)
- [Rust | Module] `page_clustering` (line 0, pub)

## File: MMSB/src/05_adaptive/page_clustering.rs

- Layer(s): 05_adaptive
- Language coverage: Rust (5)
- Element types: Function (1), Impl (1), Module (1), Struct (2)
- Total elements: 5

### Elements

- [Rust | Struct] `PageCluster` (line 0, pub)
  - Signature: `# [doc = " Cluster of pages that should be co-located"] # [derive (Debug , Clone)] pub struct PageCluster { # [doc = ...`
- [Rust | Struct] `PageClusterer` (line 0, pub)
  - Signature: `# [doc = " Page clustering engine"] # [derive (Debug)] pub struct PageClusterer { # [doc = " Current clusters"] clust...`
- [Rust | Impl] `impl PageClusterer { pub fn new (min_cluster_size : usize) -> Self { Self { clusters : Vec :: new () , min_cluster_size , } } # [doc = " Cluster pages based on co-access patterns"] pub fn cluster_pages (& mut self , coaccesses : & HashMap < (PageId , PageId) , u64 >) { self . clusters . clear () ; let mut affinities : HashMap < PageId , HashMap < PageId , u64 > > = HashMap :: new () ; for ((p1 , p2) , freq) in coaccesses { affinities . entry (* p1) . or_default () . insert (* p2 , * freq) ; affinities . entry (* p2) . or_default () . insert (* p1 , * freq) ; } let mut unclustered : HashSet < PageId > = affinities . keys () . copied () . collect () ; while ! unclustered . is_empty () { let seed = unclustered . iter () . max_by_key (| & p | { affinities . get (p) . map (| adj | adj . values () . sum :: < u64 > ()) . unwrap_or (0) }) . copied () . unwrap () ; let mut cluster = HashSet :: new () ; cluster . insert (seed) ; unclustered . remove (& seed) ; if let Some (neighbors) = affinities . get (& seed) { let mut candidates : Vec < _ > = neighbors . iter () . filter (| (p , _) | unclustered . contains (p)) . collect () ; candidates . sort_by_key (| (_ , freq) | std :: cmp :: Reverse (* * freq)) ; for (& neighbor , _) in candidates . iter () . take (self . min_cluster_size - 1) { cluster . insert (neighbor) ; unclustered . remove (& neighbor) ; } } let hotness = cluster . iter () . filter_map (| p | affinities . get (p)) . flat_map (| adj | adj . values ()) . sum () ; self . clusters . push (PageCluster { pages : cluster , hotness }) ; } self . clusters . sort_by_key (| c | std :: cmp :: Reverse (c . hotness)) ; } pub fn clusters (& self) -> & [PageCluster] { & self . clusters } } . self_ty` (line 0, priv)
- [Rust | Function] `test_page_clustering` (line 0, priv)
  - Signature: `# [test] fn test_page_clustering () { let mut clusterer = PageClusterer :: new (2) ; let mut coaccesses = HashMap :: ...`
  - Calls: PageClusterer::new, HashMap::new, insert, insert, insert, cluster_pages
- [Rust | Module] `tests` (line 0, priv)

