# Structure Group: tests

## File: MMSB/tests/benchmark_01_replay.rs

- Layer(s): root
- Language coverage: Rust (2)
- Element types: Function (2)
- Total elements: 2

### Elements

- [Rust | Function] `replay_validator_divergence_under_threshold` (line 0, priv)
  - Signature: `# [test] fn replay_validator_divergence_under_threshold () { let path = temp_log_path () ; let log = TransactionLog :...`
  - Calls: temp_log_path, unwrap, TransactionLog::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, ReplayValidator::new, unwrap, record_checkpoint, unwrap, validate_allocator, ok, std::fs::remove_file
- [Rust | Function] `temp_log_path` (line 0, priv)
  - Signature: `fn temp_log_path () -> PathBuf { let mut path = std :: env :: temp_dir () ; path . push (format ! ("mmsb_replay_{}.lo...`
  - Calls: std::env::temp_dir, push

## File: MMSB/tests/benchmark_02_integrity.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `integrity_checker_accepts_valid_delta` (line 0, priv)
  - Signature: `# [test] fn integrity_checker_accepts_valid_delta () { let registry = Arc :: new (DeviceBufferRegistry :: default ())...`
  - Calls: Arc::new, DeviceBufferRegistry::default, Arc::new, unwrap, Page::new, PageID, insert, DeltaID, PageID, Epoch, Source, into, DeltaIntegrityChecker::new, Arc::clone, validate

## File: MMSB/tests/benchmark_03_graph.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `graph_validator_detects_no_cycles` (line 0, priv)
  - Signature: `# [test] fn graph_validator_detects_no_cycles () { let graph = ShadowPageGraph :: default () ; graph . add_edge (Page...`
  - Calls: ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, GraphValidator::new, detect_cycles

## File: MMSB/tests/benchmark_04_purity.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `purity_validator_covers_semiring_operations` (line 0, priv)
  - Signature: `# [test] fn purity_validator_covers_semiring_operations () { let validator = PurityValidator :: default () ; let bool...`
  - Calls: PurityValidator::default

## File: MMSB/tests/benchmark_05_throughput.rs

- Layer(s): root
- Language coverage: Rust (2)
- Element types: Function (2)
- Total elements: 2

### Elements

- [Rust | Function] `make_delta` (line 0, priv)
  - Signature: `fn make_delta (id : u64 , page : u64) -> Delta { Delta { delta_id : DeltaID (id) , page_id : PageID (page) , epoch : ...`
  - Calls: DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `throughput_engine_exceeds_minimum_rate` (line 0, priv)
  - Signature: `# [test] fn throughput_engine_exceeds_minimum_rate () { let allocator = Arc :: new (PageAllocator :: new (PageAllocat...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, collect, map, make_delta, unwrap, process_parallel

## File: MMSB/tests/benchmark_06_tick_latency.rs

- Layer(s): root
- Language coverage: Rust (2)
- Element types: Function (2)
- Total elements: 2

### Elements

- [Rust | Function] `delta` (line 0, priv)
  - Signature: `fn delta (id : u64 , page : u64) -> Delta { Delta { delta_id : DeltaID (id) , page_id : PageID (page) , epoch : Epoch...`
  - Calls: DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `tick_latency_stays_within_budget` (line 0, priv)
  - Signature: `# [test] fn tick_latency_stays_within_budget () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConf...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, Arc::new, ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, Arc::new, MemoryMonitor::with_config, Arc::clone, MemoryMonitorConfig::default, TickOrchestrator::new, collect, map, delta, unwrap, execute_tick

## File: MMSB/tests/benchmark_07_memory.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `memory_monitor_enforces_limits` (line 0, priv)
  - Signature: `# [test] fn memory_monitor_enforces_limits () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, MemoryMonitor::with_config, Arc::clone, snapshot, unwrap, trigger_incremental_gc

## File: MMSB/tests/benchmark_08_invariants.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `invariant_checker_reports_success` (line 0, priv)
  - Signature: `# [test] fn invariant_checker_reports_success () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorCon...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, ShadowPageGraph::default, Some, Some, InvariantChecker::with_builtins, run

## File: MMSB/tests/benchmark_09_stability.rs

- Layer(s): root
- Language coverage: Rust (5)
- Element types: Function (3), Impl (1), Struct (1)
- Total elements: 5

### Elements

- [Rust | Struct] `NoiseRng` (line 0, priv)
  - Signature: `struct NoiseRng (u64) ;`
- [Rust | Function] `divergence` (line 0, priv)
  - Signature: `fn divergence (a : & [f64] , b : & [f64]) -> f64 { a . iter () . zip (b) . map (| (x , y) | (x - y) . powi (2)) . sum...`
  - Calls: sqrt, sum, map, zip, iter, powi
- [Rust | Impl] `impl NoiseRng { fn new (seed : u64) -> Self { Self (seed . max (1)) } fn next_f64 (& mut self) -> f64 { self . 0 = self . 0 . wrapping_mul (6364136223846793005) . wrapping_add (1) ; ((self . 0 >> 33) as f64) / (u32 :: MAX as f64) } fn gaussian (& mut self , std_dev : f64) -> f64 { let u1 = self . next_f64 () . max (f64 :: MIN_POSITIVE) ; let u2 = self . next_f64 () ; let mag = (- 2.0 * u1 . ln ()) . sqrt () * std_dev ; mag * (2.0 * std :: f64 :: consts :: PI * u2) . cos () } } . self_ty` (line 0, priv)
- [Rust | Function] `simulate` (line 0, priv)
  - Signature: `fn simulate (state : & [f64] , drift : f64) -> Vec < f64 > { state . iter () . enumerate () . map (| (idx , value) | ...`
  - Calls: collect, map, enumerate, iter, sin, tanh
- [Rust | Function] `stability_resists_small_noise` (line 0, priv)
  - Signature: `# [test] fn stability_resists_small_noise () { let mut rng = NoiseRng :: new (42) ; let mut baseline = vec ! [0.0 ; 6...`
  - Calls: NoiseRng::new, clone, simulate, gaussian, simulate, divergence, max

## File: MMSB/tests/benchmark_10_provenance.rs

- Layer(s): root
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `provenance_tracker_resolves_with_cache` (line 0, priv)
  - Signature: `# [test] fn provenance_tracker_resolves_with_cache () { let graph = Arc :: new (ShadowPageGraph :: default ()) ; grap...`
  - Calls: Arc::new, ShadowPageGraph::default, add_edge, PageID, PageID, add_edge, PageID, PageID, add_edge, PageID, PageID, ProvenanceTracker::with_capacity, Arc::clone, resolve, PageID, resolve, PageID

## File: MMSB/tests/delta_validation.rs

- Layer(s): root
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `dense_delta` (line 0, priv)
  - Signature: `fn dense_delta (payload : Vec < u8 > , mask : Vec < bool >) -> Delta { Delta { delta_id : DeltaID (1) , page_id : Pag...`
  - Calls: DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `rejects_mismatched_dense_lengths` (line 0, priv)
  - Signature: `# [test] fn rejects_mismatched_dense_lengths () { let delta = dense_delta (vec ! [1 , 2] , vec ! [true , true , true]...`
  - Calls: dense_delta
- [Rust | Function] `validates_dense_lengths` (line 0, priv)
  - Signature: `# [test] fn validates_dense_lengths () { let delta = dense_delta (vec ! [1 , 2 , 3] , vec ! [true , true , true]) ; a...`
  - Calls: dense_delta

## File: MMSB/tests/examples_basic.rs

- Layer(s): root
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `example_checkpoint` (line 0, priv)
  - Signature: `# [test] fn example_checkpoint () { assert ! (true) ; } . sig`
- [Rust | Function] `example_delta_operations` (line 0, priv)
  - Signature: `# [test] fn example_delta_operations () { let data = vec ! [42u8 ; 128] ; let mask = vec ! [true ; 128] ; let delta =...`
  - Calls: Delta::new_dense, DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `example_page_allocation` (line 0, priv)
  - Signature: `# [test] fn example_page_allocation () { let config = PageAllocatorConfig :: default () ; let allocator = PageAllocat...`
  - Calls: PageAllocatorConfig::default, PageAllocator::new, PageID, allocate_raw, Some, free

## File: MMSB/tests/mmsb_tests.rs

- Layer(s): root
- Language coverage: Rust (10)
- Element types: Function (10)
- Total elements: 10

### Elements

- [Rust | Function] `read_page` (line 0, priv)
  - Signature: `fn read_page (page : & Page) -> Vec < u8 > { page . data_slice () . to_vec () } . sig`
  - Calls: to_vec, data_slice
- [Rust | Function] `test_api_public_interface` (line 0, priv)
  - Signature: `# [test] # [ignore = "Julia API not available"] fn test_api_public_interface () { } . sig`
- [Rust | Function] `test_checkpoint_log_and_restore` (line 0, priv)
  - Signature: `# [test] fn test_checkpoint_log_and_restore () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, TransactionLog::new, to_string, unwrap, allocate_raw, PageID, copy_from_slice, len, copy_from_slice, data_mut_slice, unwrap, write_checkpoint, to_string, PageAllocator::new, PageAllocatorConfig::default, unwrap, TransactionLog::new, to_string, unwrap, load_checkpoint, expect, acquire_page, std::fs::remove_file, std::fs::remove_file, std::fs::remove_file
- [Rust | Function] `test_dense_delta_application` (line 0, priv)
  - Signature: `# [test] fn test_dense_delta_application () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default ()...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, Delta::new_dense, DeltaID, Epoch, Source, into, unwrap, apply_delta
- [Rust | Function] `test_gpu_delta_kernels` (line 0, priv)
  - Signature: `# [test] # [ignore = "GPU not yet implemented"] fn test_gpu_delta_kernels () { } . sig`
- [Rust | Function] `test_invalid_page_deletion_is_safe` (line 0, priv)
  - Signature: `# [test] fn test_invalid_page_deletion_is_safe () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defa...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, release, PageID
- [Rust | Function] `test_page_info_metadata_roundtrip` (line 0, priv)
  - Signature: `# [test] fn test_page_info_metadata_roundtrip () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defau...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, set_metadata, page_infos
- [Rust | Function] `test_page_snapshot_and_restore` (line 0, priv)
  - Signature: `# [test] fn test_page_snapshot_and_restore () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default ...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, fill, data_mut_slice, snapshot_pages, PageAllocator::new, PageAllocatorConfig::default, unwrap, restore_from_snapshot, unwrap, acquire_page
- [Rust | Function] `test_sparse_delta_application` (line 0, priv)
  - Signature: `# [test] fn test_sparse_delta_application () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default (...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, Delta::new_sparse, DeltaID, Epoch, Source, into, unwrap, apply_delta
- [Rust | Function] `test_thread_safe_allocator` (line 0, priv)
  - Signature: `# [test] fn test_thread_safe_allocator () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: ...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, collect, map, Arc::clone, std::thread::spawn, unwrap, allocate_raw, PageID, data_mut_slice, unwrap, join

## File: MMSB/tests/stress_memory.rs

- Layer(s): root
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `fragmentation_probe_remains_stable` (line 0, priv)
  - Signature: `# [test] fn fragmentation_probe_remains_stable () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorCo...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, MemoryMonitor::with_config, Arc::clone, Vec::new, expect, allocate_raw, PageID, Some, push, PageID, len, pop, free, snapshot, snapshot
- [Rust | Function] `incremental_gc_latency_stays_within_budget` (line 0, priv)
  - Signature: `# [test] fn incremental_gc_latency_stays_within_budget () { if cfg ! (debug_assertions) { eprintln ! ("memory stress ...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, MemoryMonitor::with_config, Arc::clone, config, expect, allocate_raw, PageID, Some, snapshot, expect, trigger_incremental_gc, config, as_secs_f64
- [Rust | Function] `one_million_pages_under_1gb_full_run` (line 0, priv)
  - Signature: `# [test] # [ignore] fn one_million_pages_under_1gb_full_run () { if cfg ! (debug_assertions) { eprintln ! ("run `carg...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, Some, MemoryMonitor::new, Arc::clone, snapshot
- [Rust | Function] `sampled_allocation_projects_under_budget` (line 0, priv)
  - Signature: `# [test] fn sampled_allocation_projects_under_budget () { let allocator = Arc :: new (PageAllocator :: new (PageAlloc...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, MemoryMonitor::new, Arc::clone, expect, allocate_raw, PageID, Some, snapshot, saturating_mul

## File: MMSB/tests/stress_stability.rs

- Layer(s): root
- Language coverage: Rust (8)
- Element types: Function (4), Impl (2), Struct (2)
- Total elements: 8

### Elements

- [Rust | Struct] `Lcg` (line 0, priv)
  - Signature: `struct Lcg (u64) ;`
- [Rust | Struct] `SignalProbe` (line 0, priv)
  - Signature: `struct SignalProbe { baseline : Vec < f64 > , values : Vec < f64 > , max_divergence : f64 , last_divergence : f64 , }`
- [Rust | Function] `apply_random_deltas` (line 0, priv)
  - Signature: `fn apply_random_deltas (allocator : & Arc < PageAllocator > , rng : & mut Lcg , count : usize , next_delta : & mut u6...`
  - Calls: PageID, next_u64, max, next_in_range, next_in_range, collect, map, iter, next_u32, DeltaID, Epoch, Source, String::new, wrapping_add, expect, acquire_page, expect, apply_delta
- [Rust | Impl] `impl Lcg { fn new (seed : u64) -> Self { Self (seed) } fn next_u64 (& mut self) -> u64 { self . 0 = self . 0 . wrapping_mul (6364136223846793005) . wrapping_add (1) ; self . 0 } fn next_u32 (& mut self) -> u32 { (self . next_u64 () >> 32) as u32 } fn next_bool (& mut self) -> bool { (self . next_u64 () & 1) == 1 } fn next_in_range (& mut self , bound : usize) -> usize { if bound == 0 { return 0 ; } (self . next_u64 () % bound as u64) as usize } fn next_f64 (& mut self) -> f64 { (self . next_u64 () as f64) / (u64 :: MAX as f64) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl SignalProbe { fn new (channels : usize , max_divergence : f64) -> Self { Self { baseline : vec ! [0.0 ; channels] , values : vec ! [0.0 ; channels] , max_divergence , last_divergence : 0.0 , } } fn step (& mut self , rng : & mut Lcg) { for (value , baseline) in self . values . iter_mut () . zip (self . baseline . iter_mut ()) { let noise = (rng . next_f64 () - 0.5) * 0.02 ; * value += noise ; * baseline = 0.995 * * baseline + 0.005 * * value ; } } fn assert_within_bounds (& mut self) { for value in & self . values { assert ! (value . is_finite () , "detected non-finite value in stability probe") ; } let divergence = self . values . iter () . zip (& self . baseline) . map (| (value , baseline) | (value - baseline) . abs ()) . fold (0.0 , f64 :: max) ; self . last_divergence = divergence ; assert ! (divergence <= self . max_divergence , "divergence {divergence:.4} exceeded bound {}" , self . max_divergence) ; } fn last_divergence (& self) -> f64 { self . last_divergence } } . self_ty` (line 0, priv)
- [Rust | Function] `mutate_graph` (line 0, priv)
  - Signature: `fn mutate_graph (graph : & ShadowPageGraph , rng : & mut Lcg) { for _ in 0 .. 4 { let mut from = PageID ((rng . next_...`
  - Calls: PageID, next_u64, PageID, next_u64, PageID, std::mem::swap, next_bool, add_edge, random_edge_type, remove_edge
- [Rust | Function] `random_edge_type` (line 0, priv)
  - Signature: `fn random_edge_type (rng : & mut Lcg) -> EdgeType { match rng . next_in_range (4) { 0 => EdgeType :: Data , 1 => Edge...`
  - Calls: next_in_range
- [Rust | Function] `ten_thousand_cycles_no_violations` (line 0, priv)
  - Signature: `# [test] fn ten_thousand_cycles_no_violations () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorCon...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, Some, ShadowPageGraph::default, DeviceBufferRegistry::default, InvariantChecker::with_builtins, Lcg::new, SignalProbe::new, apply_random_deltas, mutate_graph, step, assert_within_bounds, Some, Some, Some, run

## File: MMSB/tests/stress_throughput.rs

- Layer(s): root
- Language coverage: Rust (7)
- Element types: Function (7)
- Total elements: 7

### Elements

- [Rust | Function] `assert_throughput` (line 0, priv)
  - Signature: `fn assert_throughput (processed : usize , duration : Duration , min_target : f64) { let secs = duration . as_secs_f64...`
  - Calls: as_secs_f64, abs
- [Rust | Function] `build_noop_delta` (line 0, priv)
  - Signature: `fn build_noop_delta (delta_id : u64 , page_id : PageID) -> Delta { Delta { delta_id : DeltaID (delta_id) , page_id , ...`
  - Calls: DeltaID, Epoch, Vec::new, Vec::new, Source, String::new
- [Rust | Function] `generate_delta_batch` (line 0, priv)
  - Signature: `fn generate_delta_batch (start_delta : u64 , count : usize , page_count : u64 ,) -> Vec < Delta > { (0 .. count) . ma...`
  - Calls: collect, map, build_noop_delta, PageID
- [Rust | Function] `multi_thread_10m_deltas_per_sec` (line 0, priv)
  - Signature: `# [test] fn multi_thread_10m_deltas_per_sec () { if cfg ! (debug_assertions) { eprintln ! ("stress throughput tests r...`
  - Calls: prepare_allocator, run_throughput_benchmark
- [Rust | Function] `prepare_allocator` (line 0, priv)
  - Signature: `fn prepare_allocator () -> Arc < PageAllocator > { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorCon...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, Some
- [Rust | Function] `run_throughput_benchmark` (line 0, priv)
  - Signature: `fn run_throughput_benchmark (allocator : & Arc < PageAllocator > , workers : usize , min_target : f64 ,) { let engine...`
  - Calls: ThroughputEngine::new, Arc::clone, Duration::default, min, generate_delta_batch, expect, process_parallel, is_none, Some, assert_throughput, assert_throughput
- [Rust | Function] `single_thread_1m_deltas_per_sec` (line 0, priv)
  - Signature: `# [test] fn single_thread_1m_deltas_per_sec () { if cfg ! (debug_assertions) { eprintln ! ("stress throughput tests r...`
  - Calls: prepare_allocator, run_throughput_benchmark

## File: MMSB/tests/week27_31_integration.rs

- Layer(s): root
- Language coverage: Rust (6)
- Element types: Function (6)
- Total elements: 6

### Elements

- [Rust | Function] `test_allocator_cpu_gpu_latency` (line 0, priv)
  - Signature: `# [test] fn test_allocator_cpu_gpu_latency () { let config = PageAllocatorConfig :: default () ; let allocator = Arc ...`
  - Calls: PageAllocatorConfig::default, Arc::new, PageAllocator::new, PageID, allocate_raw, Some, free
- [Rust | Function] `test_cpu_features` (line 0, priv)
  - Signature: `# [test] fn test_cpu_features () { use mmsb_core :: utility :: CpuFeatures ; let _features = CpuFeatures :: detect ()...`
  - Calls: CpuFeatures::detect
- [Rust | Function] `test_delta_merge_simd` (line 0, priv)
  - Signature: `# [test] fn test_delta_merge_simd () { use mmsb_core :: page :: merge_deltas ; let data1 : Vec < u8 > = (0 .. 64) . c...`
  - Calls: collect, unwrap, Delta::new_dense, DeltaID, PageID, Epoch, Source, into, collect, unwrap, Delta::new_dense, DeltaID, PageID, Epoch, Source, into, merge_deltas
- [Rust | Function] `test_lockfree_allocator` (line 0, priv)
  - Signature: `# [test] fn test_lockfree_allocator () { use mmsb_core :: page :: LockFreeAllocator ; let allocator = LockFreeAllocat...`
  - Calls: LockFreeAllocator::new, get_stats
- [Rust | Function] `test_propagation_queue` (line 0, priv)
  - Signature: `# [test] fn test_propagation_queue () { use mmsb_core :: propagation :: PropagationQueue ; let _queue = PropagationQu...`
  - Calls: PropagationQueue::new
- [Rust | Function] `test_semiring_operations_tropical` (line 0, priv)
  - Signature: `# [test] fn test_semiring_operations_tropical () { let _semiring = TropicalSemiring ; assert ! (true) ; } . sig`

