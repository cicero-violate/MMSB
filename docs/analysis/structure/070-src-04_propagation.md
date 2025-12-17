# Structure Group: src/04_propagation

## File: MMSB/src/04_propagation/mod.rs

- Layer(s): 04_propagation
- Language coverage: Rust (8)
- Element types: Module (8)
- Total elements: 8

### Elements

- [Rust | Module] `propagation_command_buffer` (line 0, pub)
- [Rust | Module] `propagation_engine` (line 0, pub)
- [Rust | Module] `propagation_fastpath` (line 0, pub)
- [Rust | Module] `propagation_queue` (line 0, pub)
- [Rust | Module] `ring_buffer` (line 0, pub)
- [Rust | Module] `sparse_message_passing` (line 0, pub)
- [Rust | Module] `throughput_engine` (line 0, pub)
- [Rust | Module] `tick_orchestrator` (line 0, pub)

## File: MMSB/src/04_propagation/propagation_command_buffer.rs

- Layer(s): 04_propagation
- Language coverage: Rust (1)
- Element types: Struct (1)
- Total elements: 1

### Elements

- [Rust | Struct] `PropagationCommand` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PropagationCommand { pub page_id : PageID , pub page : Arc < Page > , pub depen...`

## File: MMSB/src/04_propagation/propagation_engine.rs

- Layer(s): 04_propagation
- Language coverage: Rust (3)
- Element types: Impl (2), Struct (1)
- Total elements: 3

### Elements

- [Rust | Impl] `Default for impl Default for PropagationEngine { fn default () -> Self { Self { callbacks : RwLock :: new (HashMap :: new ()) , queue : PropagationQueue :: default () , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `PropagationEngine` (line 0, pub)
  - Signature: `pub struct PropagationEngine { callbacks : RwLock < HashMap < PageID , Callback > > , queue : PropagationQueue , }`
- [Rust | Impl] `impl PropagationEngine { pub fn register_callback (& self , page_id : PageID , callback : Callback) { self . callbacks . write () . insert (page_id , callback) ; } pub fn enqueue (& self , command : PropagationCommand) { self . queue . push (command) ; } pub fn drain (& self) { while let Some (command) = self . queue . pop () { if let Some (cb) = self . callbacks . read () . get (& command . page_id) { (* cb) (& command . page , & command . dependencies) ; } } } } . self_ty` (line 0, priv)

## File: MMSB/src/04_propagation/propagation_fastpath.rs

- Layer(s): 04_propagation
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `passthrough` (line 0, pub)
  - Signature: `# [doc = " Placeholder fast-path propagation implementation."] pub fn passthrough (_source : & Page , _target : & mut...`

## File: MMSB/src/04_propagation/propagation_queue.rs

- Layer(s): 04_propagation
- Language coverage: Rust (8)
- Element types: Function (3), Impl (3), Module (1), Struct (1)
- Total elements: 8

### Elements

- [Rust | Impl] `Debug for impl std :: fmt :: Debug for PropagationQueue { fn fmt (& self , f : & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result { f . debug_struct ("PropagationQueue") . field ("len" , & self . len ()) . field ("capacity" , & self . ring . capacity ()) . finish () } } . self_ty` (line 0, priv)
- [Rust | Impl] `Default for impl Default for PropagationQueue { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Struct] `PropagationQueue` (line 0, pub)
  - Signature: `pub struct PropagationQueue { ring : LockFreeRingBuffer < PropagationCommand > , has_work : AtomicBool , }`
- [Rust | Function] `command` (line 0, priv)
  - Signature: `fn command (id : u64) -> PropagationCommand { let page = Arc :: new (Page :: new (PageID (id) , 8 , PageLocation :: C...`
  - Calls: Arc::new, unwrap, Page::new, PageID, Vec::new
- [Rust | Function] `drain_batch_respects_bounds` (line 0, priv)
  - Signature: `# [test] fn drain_batch_respects_bounds () { let queue = PropagationQueue :: with_capacity (8) ; for i in 0 .. 6 { qu...`
  - Calls: PropagationQueue::with_capacity, push, command, drain_batch
- [Rust | Impl] `impl PropagationQueue { pub fn new () -> Self { Self :: with_capacity (DEFAULT_CAPACITY) } pub fn with_capacity (capacity : usize) -> Self { Self { ring : LockFreeRingBuffer :: new (capacity) , has_work : AtomicBool :: new (false) , } } pub fn push (& self , command : PropagationCommand) { self . push_internal (command) ; } pub fn pop (& self) -> Option < PropagationCommand > { match self . ring . try_pop () { Some (cmd) => Some (cmd) , None => { self . has_work . store (false , Ordering :: Release) ; None } } } pub fn push_batch (& self , commands : Vec < PropagationCommand >) { for command in commands { self . push_internal (command) ; } } pub fn drain_batch (& self , max_count : usize) -> Vec < PropagationCommand > { if max_count == 0 { return Vec :: new () ; } let drained = self . ring . pop_batch (max_count) ; if drained . is_empty () { self . has_work . store (false , Ordering :: Release) ; } drained } pub fn is_empty (& self) -> bool { self . ring . is_empty () } pub fn len (& self) -> usize { self . ring . len () } fn push_internal (& self , mut command : PropagationCommand) { loop { match self . ring . try_push (command) { Ok (()) => { self . has_work . store (true , Ordering :: Release) ; break ; } Err (cmd) => { command = cmd ; thread :: yield_now () ; } } } } } . self_ty` (line 0, priv)
- [Rust | Function] `queue_roundtrip` (line 0, priv)
  - Signature: `# [test] fn queue_roundtrip () { let queue = PropagationQueue :: with_capacity (8) ; for i in 0 .. 8 { queue . push (...`
  - Calls: PropagationQueue::with_capacity, push, command, unwrap, pop
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/04_propagation/ring_buffer.rs

- Layer(s): 04_propagation
- Language coverage: Rust (16)
- Element types: Function (3), Impl (9), Module (1), Struct (3)
- Total elements: 16

### Elements

- [Rust | Struct] `CacheLine` (line 0, priv)
  - Signature: `# [repr (align (64))] struct CacheLine < T > (T) ;`
  - Generics: T
- [Rust | Impl] `Deref for impl < T > std :: ops :: Deref for CacheLine < T > { type Target = T ; fn deref (& self) -> & Self :: Target { & self . 0 } } . self_ty` (line 0, priv)
- [Rust | Impl] `DerefMut for impl < T > std :: ops :: DerefMut for CacheLine < T > { fn deref_mut (& mut self) -> & mut Self :: Target { & mut self . 0 } } . self_ty` (line 0, priv)
- [Rust | Impl] `Drop for impl < T > Drop for LockFreeRingBuffer < T > { fn drop (& mut self) { while self . try_pop () . is_some () { } } } . self_ty` (line 0, priv)
- [Rust | Impl] `FromIterator for impl < T > FromIterator < T > for LockFreeRingBuffer < T > { fn from_iter < I : IntoIterator < Item = T > > (iter : I) -> Self { let items : Vec < T > = iter . into_iter () . collect () ; let buffer = LockFreeRingBuffer :: new (items . len () . max (1)) ; buffer . push_batch (items) ; buffer } } . self_ty` (line 0, priv)
- [Rust | Struct] `LockFreeRingBuffer` (line 0, pub)
  - Signature: `# [doc = " Lock-free ring buffer with cache-friendly layout."] pub struct LockFreeRingBuffer < T > { buffer : Box < [...`
  - Generics: T
- [Rust | Impl] `Send for unsafe impl < T : Send > Send for LockFreeRingBuffer < T > { } . self_ty` (line 0, priv)
- [Rust | Struct] `Slot` (line 0, priv)
  - Signature: `struct Slot < T > { sequence : AtomicUsize , value : UnsafeCell < MaybeUninit < T > > , }`
  - Generics: T
- [Rust | Impl] `Sync for unsafe impl < T : Send > Sync for LockFreeRingBuffer < T > { } . self_ty` (line 0, priv)
- [Rust | Impl] `impl < T > CacheLine < T > { fn new (value : T) -> Self { CacheLine (value) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl < T > LockFreeRingBuffer < T > { # [doc = " Create a new ring buffer with the requested capacity (rounded up to the next power of two)."] pub fn new (capacity : usize) -> Self { let cap = NonZeroUsize :: new (capacity) . expect ("capacity must be > 0") . get () . next_power_of_two () ; let slots = (0 .. cap) . map (| seq | Slot :: new (seq)) . collect :: < Vec < _ > > () . into_boxed_slice () ; Self { mask : cap - 1 , capacity : cap , buffer : slots , head : CacheLine :: new (AtomicUsize :: new (0)) , tail : CacheLine :: new (AtomicUsize :: new (0)) , } } # [doc = " Current capacity of the buffer."] pub fn capacity (& self) -> usize { self . capacity } # [doc = " Returns true if the buffer is empty."] pub fn is_empty (& self) -> bool { self . len () == 0 } # [doc = " Approximate item count."] pub fn len (& self) -> usize { self . tail . load (Ordering :: Acquire) - self . head . load (Ordering :: Acquire) } # [doc = " Try to enqueue a single element."] pub fn try_push (& self , value : T) -> Result < () , T > { let mut pos = self . tail . load (Ordering :: Relaxed) ; loop { let slot = unsafe { self . buffer . get_unchecked (pos & self . mask) } ; let seq = slot . sequence . load (Ordering :: Acquire) ; let diff = seq as isize - pos as isize ; if diff == 0 { match self . tail . compare_exchange_weak (pos , pos + 1 , Ordering :: AcqRel , Ordering :: Relaxed) { Ok (_) => { unsafe { (* slot . value . get ()) . write (value) ; } slot . sequence . store (pos + 1 , Ordering :: Release) ; return Ok (()) ; } Err (actual) => pos = actual , } } else if diff < 0 { return Err (value) ; } else { pos = self . tail . load (Ordering :: Relaxed) ; } } } # [doc = " Try to dequeue a single element."] pub fn try_pop (& self) -> Option < T > { let mut pos = self . head . load (Ordering :: Relaxed) ; loop { let slot = unsafe { self . buffer . get_unchecked (pos & self . mask) } ; let seq = slot . sequence . load (Ordering :: Acquire) ; let diff = seq as isize - (pos + 1) as isize ; if diff == 0 { match self . head . compare_exchange_weak (pos , pos + 1 , Ordering :: AcqRel , Ordering :: Relaxed) { Ok (_) => { let value = unsafe { (* slot . value . get ()) . assume_init_read () } ; slot . sequence . store (pos + self . capacity , Ordering :: Release) ; return Some (value) ; } Err (actual) => pos = actual , } } else if diff < 0 { return None ; } else { pos = self . head . load (Ordering :: Relaxed) ; } } } # [doc = " Push as many items from the iterator as possible, returning how many were accepted."] pub fn push_batch < I > (& self , iter : I) -> usize where I : IntoIterator < Item = T > , { let mut written = 0usize ; for value in iter { if self . try_push (value) . is_err () { break ; } written += 1 ; } written } # [doc = " Pop up to `max` elements and return them in FIFO order."] pub fn pop_batch (& self , max : usize) -> Vec < T > { let mut drained = Vec :: with_capacity (max) ; for _ in 0 .. max { match self . try_pop () { Some (value) => drained . push (value) , None => break , } } drained } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl < T > Slot < T > { fn new (sequence : usize) -> Self { Self { sequence : AtomicUsize :: new (sequence) , value : UnsafeCell :: new (MaybeUninit :: uninit ()) , } } } . self_ty` (line 0, priv)
- [Rust | Function] `test_basic_push_pop` (line 0, priv)
  - Signature: `# [test] fn test_basic_push_pop () { let buffer = LockFreeRingBuffer :: new (4) ; assert ! (buffer . is_empty ()) ; b...`
  - Calls: LockFreeRingBuffer::new, unwrap, try_push, unwrap, try_push
- [Rust | Function] `test_concurrent_producers_consumers` (line 0, priv)
  - Signature: `# [test] fn test_concurrent_producers_consumers () { let buffer = Arc :: new (LockFreeRingBuffer :: new (128)) ; let ...`
  - Calls: Arc::new, LockFreeRingBuffer::new, Arc::new, AtomicUsize::new, Arc::new, AtomicUsize::new, Vec::new, Arc::clone, Arc::clone, push, thread::spawn, fetch_add, is_err, try_push, thread::yield_now, Arc::clone, Arc::clone, push, thread::spawn, load, is_some, try_pop, fetch_add, thread::sleep, Duration::from_micros, unwrap, join
- [Rust | Function] `test_wraparound_behavior` (line 0, priv)
  - Signature: `# [test] fn test_wraparound_behavior () { let buffer = LockFreeRingBuffer :: new (2) ; buffer . try_push (1) . unwrap...`
  - Calls: LockFreeRingBuffer::new, unwrap, try_push, unwrap, try_push
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/04_propagation/sparse_message_passing.rs

- Layer(s): 04_propagation
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `enqueue_sparse` (line 0, pub)
  - Signature: `# [doc = " Minimal placeholder for sparse message passing."] pub fn enqueue_sparse (queue : & PropagationQueue , comm...`
  - Calls: push

## File: MMSB/src/04_propagation/throughput_engine.rs

- Layer(s): 04_propagation
- Language coverage: Rust (17)
- Element types: Enum (1), Function (8), Impl (4), Module (1), Struct (3)
- Total elements: 17

### Elements

- [Rust | Impl] `Drop for impl Drop for ThreadPool { fn drop (& mut self) { for _ in 0 .. self . workers { let _ = self . sender . send (Message :: Shutdown) ; } for handle in self . handles . drain (..) { let _ = handle . join () ; } } } . self_ty` (line 0, priv)
- [Rust | Enum] `Message` (line 0, priv)
- [Rust | Struct] `ThreadPool` (line 0, priv)
  - Signature: `struct ThreadPool { sender : mpsc :: Sender < Message > , workers : usize , handles : Vec < thread :: JoinHandle < ()...`
- [Rust | Struct] `ThroughputEngine` (line 0, pub)
  - Signature: `pub struct ThroughputEngine { allocator : Arc < PageAllocator > , pool : ThreadPool , batch_size : usize , }`
- [Rust | Struct] `ThroughputMetrics` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct ThroughputMetrics { pub processed : usize , pub duration : Duration , pub throu...`
- [Rust | Function] `applies_batches_in_parallel` (line 0, priv)
  - Signature: `# [test] fn applies_batches_in_parallel () { let allocator = Arc :: new (PageAllocator :: new (Default :: default ())...`
  - Calls: Arc::new, PageAllocator::new, Default::default, unwrap, allocate_raw, PageID, Some, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, unwrap, process_parallel, unwrap, acquire_page, PageID, unwrap, acquire_page, PageID
- [Rust | Function] `chunk_partitions` (line 0, priv)
  - Signature: `fn chunk_partitions (partitions : Vec < (PageID , Vec < usize >) > , workers : usize ,) -> Vec < Vec < (PageID , Vec ...`
  - Calls: is_empty, Vec::new, max, len, Vec::new, Vec::with_capacity, push, len, push, Vec::with_capacity, is_empty, push
- [Rust | Function] `delta_error_to_page` (line 0, priv)
  - Signature: `fn delta_error_to_page (err : DeltaError) -> PageError { match err { DeltaError :: SizeMismatch { mask_len , payload_...`
- [Rust | Impl] `impl ThreadPool { fn new (size : usize) -> Self { let (sender , receiver) = mpsc :: channel :: < Message > () ; let receiver = Arc :: new (Mutex :: new (receiver)) ; let mut handles = Vec :: with_capacity (size) ; for _ in 0 .. size { let rx = Arc :: clone (& receiver) ; handles . push (thread :: spawn (move | | loop { let message = rx . lock () . expect ("receiver poisoned") . recv () ; match message { Ok (Message :: Job (job)) => job () , Ok (Message :: Shutdown) | Err (_) => break , } })) ; } Self { sender , workers : size , handles , } } fn execute < F > (& self , job : F) where F : FnOnce () + Send + 'static , { let _ = self . sender . send (Message :: Job (Box :: new (job))) ; } fn worker_count (& self) -> usize { self . workers } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl ThroughputEngine { pub fn new (allocator : Arc < PageAllocator > , workers : usize , batch_size : usize ,) -> Self { Self { allocator , pool : ThreadPool :: new (workers . max (1)) , batch_size : batch_size . max (1) , } } pub fn process_parallel (& self , deltas : Vec < Delta > ,) -> Result < ThroughputMetrics , PageError > { let start = Instant :: now () ; if deltas . is_empty () { return Ok (ThroughputMetrics :: new (0 , Duration :: default () , self . batch_size)) ; } let batch = Arc :: new (ColumnarDeltaBatch :: from_rows (deltas)) ; let partitions = partition_by_page (& batch) ; if partitions . is_empty () { return Ok (ThroughputMetrics :: new (0 , Duration :: default () , self . batch_size)) ; } let chunks = chunk_partitions (partitions , self . pool . worker_count ()) ; let (result_tx , result_rx) = mpsc :: channel () ; for chunk in chunks { let allocator = Arc :: clone (& self . allocator) ; let columnar = Arc :: clone (& batch) ; let tx = result_tx . clone () ; self . pool . execute (move | | { let result = process_chunk (chunk , allocator , columnar) ; tx . send (result) . ok () ; }) ; } drop (result_tx) ; let mut processed = 0usize ; for result in result_rx { match result { Ok (count) => processed += count , Err (err) => return Err (err) , } } Ok (ThroughputMetrics :: new (processed , start . elapsed () , self . batch_size)) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl ThroughputMetrics { fn new (processed : usize , duration : Duration , batch_size : usize) -> Self { let throughput = if duration . as_secs_f64 () == 0.0 { processed as f64 } else { processed as f64 / duration . as_secs_f64 () } ; let batches = if batch_size == 0 { 0 } else { (processed + batch_size - 1) / batch_size } ; Self { processed , duration , throughput , batches , } } } . self_ty` (line 0, priv)
- [Rust | Function] `make_delta` (line 0, priv)
  - Signature: `fn make_delta (id : u64 , page : u64 , payload : & [u8]) -> Delta { Delta { delta_id : DeltaID (id) , page_id : PageI...`
  - Calls: DeltaID, PageID, Epoch, collect, map, iter, to_vec, Source
- [Rust | Function] `merges_multiple_deltas_per_page` (line 0, priv)
  - Signature: `# [test] fn merges_multiple_deltas_per_page () { let allocator = Arc :: new (PageAllocator :: new (Default :: default...`
  - Calls: Arc::new, PageAllocator::new, Default::default, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, unwrap, process_parallel, unwrap, acquire_page, PageID
- [Rust | Function] `partition_by_page` (line 0, priv)
  - Signature: `fn partition_by_page (batch : & ColumnarDeltaBatch) -> Vec < (PageID , Vec < usize >) > { let mut map : HashMap < Pag...`
  - Calls: HashMap::new, len, page_id_at, push, or_default, entry, collect, into_iter
- [Rust | Function] `process_chunk` (line 0, priv)
  - Signature: `fn process_chunk (chunk : Vec < (PageID , Vec < usize >) > , allocator : Arc < PageAllocator > , batch : Arc < Column...`
  - Calls: ok_or, acquire_page, PageError::PageNotFound, is_empty, delta_at, Some, map_err, merge_deltas, apply_delta, Ok
- [Rust | Function] `reports_nonzero_throughput_for_large_batches` (line 0, priv)
  - Signature: `# [test] fn reports_nonzero_throughput_for_large_batches () { let allocator = Arc :: new (PageAllocator :: new (Defau...`
  - Calls: Arc::new, PageAllocator::new, Default::default, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, Vec::new, push, make_delta, unwrap, process_parallel
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/04_propagation/tick_orchestrator.rs

- Layer(s): 04_propagation
- Language coverage: Rust (8)
- Element types: Function (4), Impl (1), Module (1), Struct (2)
- Total elements: 8

### Elements

- [Rust | Struct] `TickMetrics` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct TickMetrics { pub propagation : Duration , pub graph_validation : Duration , pu...`
- [Rust | Struct] `TickOrchestrator` (line 0, pub)
  - Signature: `pub struct TickOrchestrator { throughput : ThroughputEngine , graph : Arc < ShadowPageGraph > , memory_monitor : Arc ...`
- [Rust | Function] `gc_invoked_when_threshold_low` (line 0, priv)
  - Signature: `# [test] fn gc_invoked_when_threshold_low () { let (orchestrator , _) = orchestrator (1) ; let deltas = vec ! [sample...`
  - Calls: orchestrator, unwrap, execute_tick
- [Rust | Impl] `impl TickOrchestrator { pub fn new (throughput : ThroughputEngine , graph : Arc < ShadowPageGraph > , memory_monitor : Arc < MemoryMonitor > ,) -> Self { Self { throughput , graph , memory_monitor , tick_budget_ms : 16 , } } pub fn execute_tick (& self , deltas : Vec < Delta >) -> Result < TickMetrics , PageError > { let tick_start = Instant :: now () ; let throughput_metrics = self . throughput . process_parallel (deltas) ? ; let graph_report = { let validator = GraphValidator :: new (& self . graph) ; validator . detect_cycles () } ; let gc_metrics = self . memory_monitor . trigger_incremental_gc (self . memory_monitor . config () . incremental_batch_pages) ; let total = tick_start . elapsed () ; Ok (TickMetrics { propagation : throughput_metrics . duration , graph_validation : graph_report . duration , gc : gc_metrics . map (| m | m . duration) . unwrap_or_default () , total , throughput : throughput_metrics . throughput , gc_invoked : gc_metrics . is_some () , graph_has_cycle : graph_report . has_cycle , processed : throughput_metrics . processed , }) } pub fn budget_ms (& self) -> u64 { self . tick_budget_ms } } . self_ty` (line 0, priv)
- [Rust | Function] `orchestrator` (line 0, priv)
  - Signature: `fn orchestrator (threshold : usize) -> (TickOrchestrator , Arc < PageAllocator >) { let allocator = Arc :: new (PageA...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, ThroughputEngine::new, Arc::clone, Arc::new, ShadowPageGraph::default, add_edge, PageID, PageID, Arc::new, MemoryMonitor::with_config, Arc::clone, MemoryMonitorConfig::default, TickOrchestrator::new
- [Rust | Function] `sample_delta` (line 0, priv)
  - Signature: `fn sample_delta (id : u64 , page : u64 , value : u8) -> Delta { Delta { delta_id : DeltaID (id) , page_id : PageID (p...`
  - Calls: DeltaID, PageID, Epoch, Source
- [Rust | Module] `tests` (line 0, priv)
- [Rust | Function] `tick_metrics_capture_all_phases` (line 0, priv)
  - Signature: `# [test] fn tick_metrics_capture_all_phases () { let (orchestrator , _) = orchestrator (usize :: MAX) ; let deltas = ...`
  - Calls: orchestrator, unwrap, execute_tick

