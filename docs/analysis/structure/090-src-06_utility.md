# Structure Group: src/06_utility

## File: MMSB/src/06_utility/cpu_features.rs

- Layer(s): 06_utility
- Language coverage: Rust (5)
- Element types: Function (3), Impl (1), Struct (1)
- Total elements: 5

### Elements

- [Rust | Struct] `CpuFeatures` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Copy)] pub struct CpuFeatures { pub avx2 : bool , pub avx512f : bool , pub sse42 : bool , ...`
- [Rust | Function] `cpu_has_avx2` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_avx2 () -> bool { CpuFeatures :: get () . avx2 } . sig`
  - Calls: CpuFeatures::get
- [Rust | Function] `cpu_has_avx512` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_avx512 () -> bool { CpuFeatures :: get () . avx512f } . sig`
  - Calls: CpuFeatures::get
- [Rust | Function] `cpu_has_sse42` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_sse42 () -> bool { CpuFeatures :: get () . sse42 } . sig`
  - Calls: CpuFeatures::get
- [Rust | Impl] `impl CpuFeatures { pub fn detect () -> Self { # [cfg (target_arch = "x86_64")] { Self { avx2 : is_x86_feature_detected ! ("avx2") , avx512f : is_x86_feature_detected ! ("avx512f") , sse42 : is_x86_feature_detected ! ("sse4.2") , bmi2 : is_x86_feature_detected ! ("bmi2") , } } # [cfg (not (target_arch = "x86_64"))] { Self { avx2 : false , avx512f : false , sse42 : false , bmi2 : false , } } } pub fn get () -> & 'static CpuFeatures { CPU_FEATURES . get_or_init (Self :: detect) } } . self_ty` (line 0, priv)

## File: MMSB/src/06_utility/invariant_checker.rs

- Layer(s): 06_utility
- Language coverage: Rust (20)
- Element types: Function (5), Impl (7), Module (1), Struct (6), Trait (1)
- Total elements: 20

### Elements

- [Rust | Impl] `Default for impl Default for GraphAcyclicity { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Struct] `EpochMonotonicity` (line 0, pub)
  - Signature: `# [derive (Default)] pub struct EpochMonotonicity { seen : RwLock < HashMap < PageID , Epoch > > , }`
- [Rust | Struct] `GraphAcyclicity` (line 0, pub)
  - Signature: `pub struct GraphAcyclicity ;`
- [Rust | Trait] `Invariant` (line 0, pub)
- [Rust | Impl] `Invariant for impl Invariant for EpochMonotonicity { fn name (& self) -> & 'static str { "EpochMonotonicity" } fn check (& self , ctx : & InvariantContext) -> InvariantResult { let allocator = match ctx . allocator { Some (alloc) => alloc , None => { return InvariantResult :: fail (self . name () , "allocator unavailable") ; } } ; let mut guard = self . seen . write () ; for page in allocator . page_infos () { match guard . get (& page . page_id) { Some (epoch) if page . epoch < epoch . 0 => { return InvariantResult :: fail (self . name () , format ! ("epoch regression on page {}: {} < {}" , page . page_id . 0 , page . epoch , epoch . 0) ,) ; } _ => { guard . insert (page . page_id , Epoch (page . epoch)) ; } } } InvariantResult :: ok (self . name ()) } } . self_ty` (line 0, priv)
- [Rust | Impl] `Invariant for impl Invariant for GraphAcyclicity { fn name (& self) -> & 'static str { "GraphAcyclicity" } fn check (& self , ctx : & InvariantContext) -> InvariantResult { let graph = match ctx . graph { Some (graph) => graph , None => return InvariantResult :: fail (self . name () , "graph unavailable") , } ; let validator = GraphValidator :: new (graph) ; let report = validator . detect_cycles () ; if report . has_cycle { let ids : Vec < String > = report . cycle . iter () . map (| id | id . 0 . to_string ()) . collect () ; InvariantResult :: fail (self . name () , format ! ("cycle detected: {}" , ids . join (" → ")) ,) } else { InvariantResult :: ok (self . name ()) } } } . self_ty` (line 0, priv)
- [Rust | Impl] `Invariant for impl Invariant for PageConsistency { fn name (& self) -> & 'static str { "PageConsistency" } fn check (& self , ctx : & InvariantContext) -> InvariantResult { let allocator = match ctx . allocator { Some (alloc) => alloc , None => return InvariantResult :: fail (self . name () , "allocator unavailable") , } ; for snapshot in allocator . snapshot_pages () { if snapshot . data . len () != snapshot . size { return InvariantResult :: fail (self . name () , format ! ("page {} payload mismatch: {} != {}" , snapshot . page_id . 0 , snapshot . data . len () , snapshot . size) ,) ; } if let Err (err) = validate_metadata_blob (& snapshot . metadata_blob) { return InvariantResult :: fail (self . name () , format ! ("page {} metadata invalid: {err}" , snapshot . page_id . 0) ,) ; } } InvariantResult :: ok (self . name ()) } } . self_ty` (line 0, priv)
- [Rust | Struct] `InvariantChecker` (line 0, pub)
  - Signature: `pub struct InvariantChecker { invariants : Vec < Box < dyn Invariant > > , }`
- [Rust | Struct] `InvariantContext` (line 0, pub)
  - Signature: `# [derive (Clone)] pub struct InvariantContext < 'a > { pub allocator : Option < & 'a PageAllocator > , pub graph : O...`
  - Generics: 'a
- [Rust | Struct] `InvariantResult` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct InvariantResult { pub name : & 'static str , pub passed : bool , pub details : ...`
- [Rust | Struct] `PageConsistency` (line 0, pub)
  - Signature: `# [derive (Default)] pub struct PageConsistency ;`
- [Rust | Function] `epoch_invariant_detects_regressions` (line 0, priv)
  - Signature: `# [test] fn epoch_invariant_detects_regressions () { let allocator = PageAllocator :: new (PageAllocatorConfig :: def...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, unwrap, acquire_page, PageID, set_epoch, Epoch, Some, InvariantChecker::new, register, EpochMonotonicity::default, unwrap, acquire_page, PageID, set_epoch, Epoch
- [Rust | Function] `graph_acyclicity_detects_cycles` (line 0, priv)
  - Signature: `# [test] fn graph_acyclicity_detects_cycles () { let graph = ShadowPageGraph :: default () ; graph . add_edge (PageID...`
  - Calls: ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, Some, InvariantChecker::new, register, GraphAcyclicity::new
- [Rust | Impl] `impl GraphAcyclicity { pub fn new () -> Self { Self } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl InvariantChecker { pub fn new () -> Self { Self { invariants : Vec :: new () , } } pub fn with_builtins () -> Self { let mut checker = Self :: new () ; checker . register (EpochMonotonicity :: default ()) ; checker . register (PageConsistency :: default ()) ; checker . register (GraphAcyclicity :: new ()) ; checker } pub fn register < I > (& mut self , invariant : I) where I : Invariant + 'static , { self . invariants . push (Box :: new (invariant)) ; } pub fn run (& self , ctx : & InvariantContext < '_ >) -> Vec < InvariantResult > { self . invariants . iter () . map (| inv | inv . check (ctx)) . collect () } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl InvariantResult { pub fn ok (name : & 'static str) -> Self { Self { name , passed : true , details : None , } } pub fn fail (name : & 'static str , msg : impl Into < String >) -> Self { Self { name , passed : false , details : Some (msg . into ()) , } } } . self_ty` (line 0, priv)
- [Rust | Function] `read_bytes` (line 0, priv)
  - Signature: `fn read_bytes (blob : & [u8] , cursor : & mut usize , len : usize) -> Result < () , PageError > { if * cursor + len >...`
  - Calls: len, Err, PageError::MetadataDecode, Ok
- [Rust | Function] `read_u32` (line 0, priv)
  - Signature: `fn read_u32 (blob : & [u8] , cursor : & mut usize) -> Result < u32 , PageError > { if * cursor + 4 > blob . len () { ...`
  - Calls: len, Err, PageError::MetadataDecode, u32::from_le_bytes, unwrap, try_into, Ok
- [Rust | Module] `tests` (line 0, priv)
- [Rust | Function] `validate_metadata_blob` (line 0, priv)
  - Signature: `fn validate_metadata_blob (blob : & [u8]) -> Result < () , PageError > { if blob . is_empty () { return Ok (()) ; } l...`
  - Calls: is_empty, Ok, read_u32, read_u32, read_bytes, read_u32, read_bytes, Ok

## File: MMSB/src/06_utility/memory_monitor.rs

- Layer(s): 06_utility
- Language coverage: Rust (13)
- Element types: Function (5), Impl (2), Module (1), Struct (5)
- Total elements: 13

### Elements

- [Rust | Impl] `Default for impl Default for MemoryMonitorConfig { fn default () -> Self { Self { gc_threshold_bytes : 1 * 1024 * 1024 * 1024 , cold_page_age_limit : 3 , incremental_batch_pages : 32 , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `GCMetrics` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Copy)] pub struct GCMetrics { pub reclaimed_pages : usize , pub reclaimed_bytes : usize , ...`
- [Rust | Struct] `MemoryMonitor` (line 0, pub)
  - Signature: `pub struct MemoryMonitor { allocator : Arc < PageAllocator > , stats : Arc < AllocatorStats > , config : MemoryMonito...`
- [Rust | Struct] `MemoryMonitorConfig` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct MemoryMonitorConfig { pub gc_threshold_bytes : usize , pub cold_page_age_limit ...`
- [Rust | Struct] `MemorySnapshot` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct MemorySnapshot { pub total_pages : usize , pub total_bytes : usize , pub avg_by...`
- [Rust | Struct] `PageAging` (line 0, priv)
  - Signature: `struct PageAging { last_epoch : u32 , age : u64 , }`
- [Rust | Function] `allocator` (line 0, priv)
  - Signature: `fn allocator () -> Arc < PageAllocator > { Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ())) } . sig`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default
- [Rust | Function] `gc_trigger_depends_on_threshold` (line 0, priv)
  - Signature: `# [test] fn gc_trigger_depends_on_threshold () { let allocator = allocator () ; allocator . allocate_raw (PageID (1) ...`
  - Calls: allocator, unwrap, allocate_raw, PageID, Some, MemoryMonitor::with_config, Arc::clone, MemoryMonitor::with_config, Arc::clone
- [Rust | Impl] `impl MemoryMonitor { pub fn new (allocator : Arc < PageAllocator >) -> Self { Self :: with_config (allocator , MemoryMonitorConfig :: default ()) } pub fn with_config (allocator : Arc < PageAllocator > , config : MemoryMonitorConfig ,) -> Self { let stats = allocator . stats () ; Self { allocator , stats , config , aging : Mutex :: new (HashMap :: new ()) , } } pub fn snapshot (& self) -> MemorySnapshot { let infos = self . allocator . page_infos () ; self . build_snapshot (& infos) } pub fn trigger_incremental_gc (& self , budget_pages : usize ,) -> Option < GCMetrics > { let infos = self . allocator . page_infos () ; let snapshot = self . build_snapshot (& infos) ; if snapshot . total_bytes <= self . config . gc_threshold_bytes && snapshot . cold_pages . is_empty () { return None ; } let mut info_map = HashMap :: new () ; for info in & infos { info_map . insert (info . page_id , info . size) ; } let target = if budget_pages == 0 { self . config . incremental_batch_pages } else { budget_pages . min (self . config . incremental_batch_pages . max (1)) } ; let mut reclaimed_pages = 0usize ; let mut reclaimed_bytes = 0usize ; let start = Instant :: now () ; for page_id in snapshot . cold_pages . into_iter () . take (target) { if let Some (bytes) = info_map . get (& page_id) . copied () { self . allocator . free (page_id) ; self . aging . lock () . remove (& page_id) ; reclaimed_pages += 1 ; reclaimed_bytes += bytes ; } } if reclaimed_pages == 0 { return None ; } Some (GCMetrics { reclaimed_pages , reclaimed_bytes , duration : start . elapsed () , }) } pub fn stats (& self) -> (u64 , u64) { self . stats . snapshot () } pub fn config (& self) -> & MemoryMonitorConfig { & self . config } fn build_snapshot (& self , infos : & [crate :: page :: PageInfo]) -> MemorySnapshot { let total_pages = infos . len () ; let total_bytes : usize = infos . iter () . map (| info | info . size) . sum () ; let avg_bytes = if total_pages == 0 { 0 } else { total_bytes / total_pages } ; let cold_pages = self . update_aging (infos) ; MemorySnapshot { total_pages , total_bytes , avg_bytes_per_page : avg_bytes , cold_pages , collected_at : Instant :: now () , } } fn update_aging (& self , infos : & [crate :: page :: PageInfo] ,) -> Vec < PageID > { let mut aging = self . aging . lock () ; let mut cold = Vec :: new () ; let mut seen = HashSet :: new () ; for info in infos { seen . insert (info . page_id) ; let entry = aging . entry (info . page_id) . or_insert (PageAging { last_epoch : info . epoch , age : 0 , }) ; if entry . last_epoch == info . epoch { entry . age += 1 ; } else { entry . last_epoch = info . epoch ; entry . age = 0 ; } if entry . age > self . config . cold_page_age_limit { cold . push (info . page_id) ; } } aging . retain (| page_id , _ | seen . contains (page_id)) ; cold } } . self_ty` (line 0, priv)
- [Rust | Function] `incremental_gc_reclaims_pages_under_budget` (line 0, priv)
  - Signature: `# [test] fn incremental_gc_reclaims_pages_under_budget () { let allocator = allocator () ; for id in 1 ..= 4 { alloca...`
  - Calls: allocator, unwrap, allocate_raw, PageID, Some, MemoryMonitor::with_config, Arc::clone, unwrap, trigger_incremental_gc
- [Rust | Function] `snapshot_identifies_cold_pages` (line 0, priv)
  - Signature: `# [test] fn snapshot_identifies_cold_pages () { let allocator = allocator () ; allocator . allocate_raw (PageID (1) ,...`
  - Calls: allocator, unwrap, allocate_raw, PageID, Some, MemoryMonitor::with_config, Arc::clone, snapshot, snapshot, unwrap, acquire_page, PageID, set_epoch, Epoch, snapshot
- [Rust | Function] `snapshot_reflects_allocator_state` (line 0, priv)
  - Signature: `# [test] fn snapshot_reflects_allocator_state () { let allocator = allocator () ; allocator . allocate_raw (PageID (1...`
  - Calls: allocator, unwrap, allocate_raw, PageID, Some, unwrap, allocate_raw, PageID, Some, MemoryMonitor::new, Arc::clone, snapshot
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/06_utility/mod.rs

- Layer(s): 06_utility
- Language coverage: Rust (5)
- Element types: Module (5)
- Total elements: 5

### Elements

- [Rust | Module] `cpu_features` (line 0, pub)
- [Rust | Module] `invariant_checker` (line 0, pub)
- [Rust | Module] `memory_monitor` (line 0, pub)
- [Rust | Module] `provenance_tracker` (line 0, pub)
- [Rust | Module] `telemetry` (line 0, pub)

## File: MMSB/src/06_utility/provenance_tracker.rs

- Layer(s): 06_utility
- Language coverage: Rust (6)
- Element types: Function (2), Impl (1), Module (1), Struct (2)
- Total elements: 6

### Elements

- [Rust | Struct] `ProvenanceResult` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct ProvenanceResult { pub chain : Vec < PageID > , pub duration : std :: time :: D...`
- [Rust | Struct] `ProvenanceTracker` (line 0, pub)
  - Signature: `pub struct ProvenanceTracker { graph : Arc < ShadowPageGraph > , cache : parking_lot :: Mutex < HashMap < PageID , Ve...`
- [Rust | Function] `cache_does_not_grow_unbounded` (line 0, priv)
  - Signature: `# [test] fn cache_does_not_grow_unbounded () { let graph = Arc :: new (ShadowPageGraph :: default ()) ; for id in 1 ....`
  - Calls: Arc::new, ShadowPageGraph::default, add_edge, PageID, PageID, ProvenanceTracker::with_capacity, Arc::clone, resolve, PageID
- [Rust | Impl] `impl ProvenanceTracker { pub fn new (graph : Arc < ShadowPageGraph >) -> Self { Self :: with_capacity (graph , 128 , 32) } pub fn with_capacity (graph : Arc < ShadowPageGraph > , capacity : usize , depth_limit : usize ,) -> Self { Self { graph , cache : parking_lot :: Mutex :: new (HashMap :: new ()) , order : parking_lot :: Mutex :: new (VecDeque :: new ()) , capacity : capacity . max (4) , depth_limit : depth_limit . max (1) , } } pub fn resolve (& self , page_id : PageID) -> ProvenanceResult { let start = Instant :: now () ; if let Some (chain) = self . cache . lock () . get (& page_id) . cloned () { return ProvenanceResult { chain , duration : start . elapsed () , from_cache : true , } ; } let chain = self . resolve_uncached (page_id) ; self . insert_cache (page_id , chain . clone ()) ; ProvenanceResult { chain , duration : start . elapsed () , from_cache : false , } } fn resolve_uncached (& self , page_id : PageID) -> Vec < PageID > { let adjacency = self . graph . adjacency . read () . clone () ; let mut reverse : HashMap < PageID , Vec < PageID > > = HashMap :: new () ; for (from , edges) in adjacency . iter () { for (to , _) in edges { reverse . entry (* to) . or_default () . push (* from) ; } } let mut chain = Vec :: new () ; let mut current = page_id ; chain . push (current) ; for _ in 0 .. self . depth_limit { if let Some (parents) = reverse . get (& current) { if let Some (parent) = parents . first () { current = * parent ; chain . push (current) ; } else { break ; } } else { break ; } } chain } fn insert_cache (& self , key : PageID , chain : Vec < PageID >) { let mut cache = self . cache . lock () ; let mut order = self . order . lock () ; if cache . contains_key (& key) { order . retain (| id | id != & key) ; } cache . insert (key , chain) ; order . push_front (key) ; while order . len () > self . capacity { if let Some (evicted) = order . pop_back () { cache . remove (& evicted) ; } } } } . self_ty` (line 0, priv)
- [Rust | Function] `resolves_chain_with_depth_limit` (line 0, priv)
  - Signature: `# [test] fn resolves_chain_with_depth_limit () { let graph = Arc :: new (ShadowPageGraph :: default ()) ; graph . add...`
  - Calls: Arc::new, ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, add_edge, PageID, PageID, ProvenanceTracker::with_capacity, Arc::clone, resolve, PageID, resolve, PageID
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/06_utility/telemetry.rs

- Layer(s): 06_utility
- Language coverage: Rust (9)
- Element types: Function (3), Impl (3), Module (1), Struct (2)
- Total elements: 9

### Elements

- [Rust | Impl] `Default for impl Default for Telemetry { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Struct] `Telemetry` (line 0, pub)
  - Signature: `# [doc = " Telemetry counters for system metrics"] # [derive (Debug)] pub struct Telemetry { # [doc = " Total cache m...`
- [Rust | Struct] `TelemetrySnapshot` (line 0, pub)
  - Signature: `# [doc = " Snapshot of telemetry metrics"] # [derive (Debug , Clone , Copy)] pub struct TelemetrySnapshot { pub cache...`
- [Rust | Impl] `impl Telemetry { # [doc = " Create new telemetry tracker"] pub fn new () -> Self { Self { cache_misses : AtomicU64 :: new (0) , cache_hits : AtomicU64 :: new (0) , allocations : AtomicU64 :: new (0) , bytes_allocated : AtomicU64 :: new (0) , propagations : AtomicU64 :: new (0) , propagation_latency_us : AtomicU64 :: new (0) , start_time : Instant :: now () , } } # [doc = " Record cache miss"] pub fn record_cache_miss (& self) { self . cache_misses . fetch_add (1 , Ordering :: Relaxed) ; } # [doc = " Record cache hit"] pub fn record_cache_hit (& self) { self . cache_hits . fetch_add (1 , Ordering :: Relaxed) ; } # [doc = " Record allocation"] pub fn record_allocation (& self , bytes : u64) { self . allocations . fetch_add (1 , Ordering :: Relaxed) ; self . bytes_allocated . fetch_add (bytes , Ordering :: Relaxed) ; } # [doc = " Record propagation"] pub fn record_propagation (& self , latency_us : u64) { self . propagations . fetch_add (1 , Ordering :: Relaxed) ; self . propagation_latency_us . fetch_add (latency_us , Ordering :: Relaxed) ; } # [doc = " Get current snapshot"] pub fn snapshot (& self) -> TelemetrySnapshot { TelemetrySnapshot { cache_misses : self . cache_misses . load (Ordering :: Relaxed) , cache_hits : self . cache_hits . load (Ordering :: Relaxed) , allocations : self . allocations . load (Ordering :: Relaxed) , bytes_allocated : self . bytes_allocated . load (Ordering :: Relaxed) , propagations : self . propagations . load (Ordering :: Relaxed) , propagation_latency_us : self . propagation_latency_us . load (Ordering :: Relaxed) , elapsed_ms : self . start_time . elapsed () . as_millis () as u64 , } } # [doc = " Reset all counters"] pub fn reset (& self) { self . cache_misses . store (0 , Ordering :: Relaxed) ; self . cache_hits . store (0 , Ordering :: Relaxed) ; self . allocations . store (0 , Ordering :: Relaxed) ; self . bytes_allocated . store (0 , Ordering :: Relaxed) ; self . propagations . store (0 , Ordering :: Relaxed) ; self . propagation_latency_us . store (0 , Ordering :: Relaxed) ; } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl TelemetrySnapshot { # [doc = " Compute cache hit rate"] pub fn cache_hit_rate (& self) -> f64 { let total = self . cache_hits + self . cache_misses ; if total == 0 { 0.0 } else { self . cache_hits as f64 / total as f64 } } # [doc = " Compute average propagation latency"] pub fn avg_propagation_latency_us (& self) -> f64 { if self . propagations == 0 { 0.0 } else { self . propagation_latency_us as f64 / self . propagations as f64 } } # [doc = " Compute memory overhead (bytes per allocation)"] pub fn avg_allocation_size (& self) -> f64 { if self . allocations == 0 { 0.0 } else { self . bytes_allocated as f64 / self . allocations as f64 } } } . self_ty` (line 0, priv)
- [Rust | Function] `test_cache_hit_rate` (line 0, priv)
  - Signature: `# [test] fn test_cache_hit_rate () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemetr...`
  - Calls: Telemetry::new, record_cache_hit, record_cache_hit, record_cache_hit, record_cache_miss, snapshot
- [Rust | Function] `test_reset` (line 0, priv)
  - Signature: `# [test] fn test_reset () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemetry . reset...`
  - Calls: Telemetry::new, record_cache_hit, reset, snapshot
- [Rust | Function] `test_telemetry_basic` (line 0, priv)
  - Signature: `# [test] fn test_telemetry_basic () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemet...`
  - Calls: Telemetry::new, record_cache_hit, record_cache_miss, record_allocation, snapshot
- [Rust | Module] `tests` (line 0, priv)

