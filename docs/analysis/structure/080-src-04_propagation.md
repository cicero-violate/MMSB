# Structure Group: src/04_propagation

## File: MMSB/src/04_propagation/PropagationEngine.jl

- Layer(s): 04_propagation
- Language coverage: Julia (25)
- Element types: Function (22), Module (1), Struct (2)
- Total elements: 25

### Elements

- [Julia | Module] `PropagationEngine` (line 8, pub)
- [Julia | Struct] `CUDAGraphState` (line 35, pub)
  - Signature: `mutable struct CUDAGraphState`
- [Julia | Struct] `PropagationQueue` (line 44, pub)
  - Signature: `mutable struct PropagationQueue`
- [Julia | Function] `enable_graph_capture` (line 57, pub)
  - Signature: `enable_graph_capture(state::MMSBState)`
  - Calls: CUDAGraphState, get!
- [Julia | Function] `disable_graph_capture` (line 69, pub)
  - Signature: `disable_graph_capture(state::MMSBState)`
  - Calls: ccall, haskey
- [Julia | Function] `replay_cuda_graph` (line 90, pub)
  - Signature: `replay_cuda_graph(state::MMSBState, stream::Ptr{Cvoid})`
  - Calls: ccall, haskey
- [Julia | Function] `batch_route_deltas!` (line 111, pub)
  - Signature: `batch_route_deltas!(state::MMSBState, deltas::Vector{DeltaType})`
  - Calls: haskey, isempty, push!, route_delta!
- [Julia | Function] `_buffer` (line 146, pub)
  - Signature: `_buffer(state::MMSBState)::PropagationQueue`
  - Calls: PropagationQueue, get!
- [Julia | Function] `register_recompute_fn!` (line 158, pub)
  - Signature: `register_recompute_fn!(state::MMSBState, page_id::PageID, fn::Function)`
  - Calls: PageNotFoundError, UInt64, get_page, throw
- [Julia | Function] `register_passthrough_recompute!` (line 171, pub)
  - Signature: `register_passthrough_recompute!(state::MMSBState, target_page_id::PageID, source_page_id::PageID; transform`
- [Julia | Function] `queue_recomputation!` (line 185, pub)
  - Signature: `queue_recomputation!(state::MMSBState, page_id::PageID)`
  - Calls: _buffer, push!
- [Julia | Function] `propagate_change!` (line 198, pub)
  - Signature: `propagate_change!(state::MMSBState, changed_page_id::PageID, mode::PropagationMode`
- [Julia | Function] `propagate_change!` (line 203, pub)
  - Signature: `propagate_change!(state::MMSBState, changed_pages::AbstractVector{PageID}, mode::PropagationMode`
- [Julia | Function] `_aggregate_children` (line 213, pub)
  - Signature: `_aggregate_children(state::MMSBState, parents::AbstractVector{PageID})`
  - Calls: get!, get_children, push!
- [Julia | Function] `_execute_command_buffer!` (line 226, pub)
  - Signature: `_execute_command_buffer!(state::MMSBState, commands::Dict{PageID, Set{EdgeType}}, mode::PropagationMode)`
  - Calls: _apply_edges!
- [Julia | Function] `_apply_edges!` (line 234, pub)
  - Signature: `_apply_edges!(state::MMSBState, page_id::PageID, edges::Set{EdgeType}, mode::PropagationMode)`
  - Calls: _handle_data_dependency!, delete!, invalidate_compilation!, mark_page_stale!, schedule_gpu_sync!
- [Julia | Function] `_handle_data_dependency!` (line 247, pub)
  - Signature: `_handle_data_dependency!(state::MMSBState, page_id::PageID, mode::PropagationMode)`
  - Calls: emit_event!, queue_recomputation!, recompute_page!
- [Julia | Function] `_collect_descendants` (line 260, pub)
  - Signature: `_collect_descendants(state::MMSBState, page_id::PageID)::Set{PageID}`
  - Calls: get_children, isempty, popfirst!, push!
- [Julia | Function] `schedule_propagation!` (line 280, pub)
  - Signature: `schedule_propagation!(state::MMSBState, changed_pages::Vector{PageID})`
  - Calls: _collect_descendants, collect, queue_recomputation!, topological_order_subset, union!
- [Julia | Function] `execute_propagation!` (line 296, pub)
  - Signature: `execute_propagation!(state::MMSBState)`
  - Calls: _buffer, delete!, isempty, popfirst!, recompute_page!
- [Julia | Function] `recompute_page!` (line 310, pub)
  - Signature: `recompute_page!(state::MMSBState, page_id::PageID)`
  - Calls: InvalidDeltaError, UInt64, eachindex, get, get_page, length, read_page, recompute_fn, throw
- [Julia | Function] `mark_page_stale!` (line 334, pub)
  - Signature: `mark_page_stale!(state::MMSBState, page_id::PageID)`
  - Calls: emit_event!, get_page
- [Julia | Function] `schedule_gpu_sync!` (line 345, pub)
  - Signature: `schedule_gpu_sync!(state::MMSBState, page_id::PageID)`
  - Calls: emit_event!, get_page
- [Julia | Function] `invalidate_compilation!` (line 355, pub)
  - Signature: `invalidate_compilation!(state::MMSBState, page_id::PageID)`
  - Calls: emit_event!, get_page
- [Julia | Function] `topological_order_subset` (line 368, pub)
  - Signature: `topological_order_subset(state::MMSBState, subset::Vector{PageID})::Vector{PageID}`
  - Calls: Set, get, get_children, isempty, popfirst!, push!

## File: MMSB/src/04_propagation/PropagationScheduler.jl

- Layer(s): 04_propagation
- Language coverage: Julia (2)
- Element types: Function (1), Module (1)
- Total elements: 2

### Elements

- [Julia | Module] `PropagationScheduler` (line 1, pub)
- [Julia | Function] `schedule!` (line 8, pub)
  - Signature: `schedule!(engine, commands)`
  - Calls: engine.drain, engine.enqueue

## File: MMSB/src/04_propagation/TransactionIsolation.jl

- Layer(s): 04_propagation
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `TransactionIsolation` (line 1, pub)
- [Julia | Struct] `Transaction` (line 5, pub)
  - Signature: `mutable struct Transaction`
- [Julia | Function] `begin_transaction` (line 11, pub)
  - Signature: `begin_transaction(s)`
  - Calls: Transaction, rand
- [Julia | Function] `commit_transaction` (line 12, pub)
  - Signature: `commit_transaction(s)`
  - Calls: delete!, haskey
- [Julia | Function] `rollback_transaction` (line 13, pub)
  - Signature: `rollback_transaction(s)`
  - Calls: delete!, haskey
- [Julia | Function] `with_transaction` (line 14, pub)
  - Signature: `with_transaction(f,s)`
  - Calls: begin_transaction, commit_transaction, f, rethrow, rollback_transaction

## File: MMSB/src/04_propagation/mod.rs

- Layer(s): 04_propagation
- Language coverage: Rust (5)
- Element types: Module (5)
- Total elements: 5

### Elements

- [Rust | Module] `propagation_command_buffer` (line 0, pub)
- [Rust | Module] `propagation_engine` (line 0, pub)
- [Rust | Module] `propagation_fastpath` (line 0, pub)
- [Rust | Module] `propagation_queue` (line 0, pub)
- [Rust | Module] `sparse_message_passing` (line 0, pub)

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
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `PropagationQueue` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct PropagationQueue { queue : Mutex < VecDeque < PropagationCommand > > , has_wo...`
- [Rust | Impl] `impl PropagationQueue { pub fn new () -> Self { Self { queue : Mutex :: new (VecDeque :: new ()) , has_work : Arc :: new (AtomicBool :: new (false)) , } } pub fn push (& self , command : PropagationCommand) { self . queue . lock () . push_back (command) ; self . has_work . store (true , Ordering :: Release) ; } pub fn pop (& self) -> Option < PropagationCommand > { let result = self . queue . lock () . pop_front () ; if result . is_none () { self . has_work . store (false , Ordering :: Release) ; } result } pub fn push_batch (& self , commands : Vec < PropagationCommand >) { let mut queue = self . queue . lock () ; queue . extend (commands) ; self . has_work . store (! queue . is_empty () , Ordering :: Release) ; } pub fn drain_batch (& self , max_count : usize) -> Vec < PropagationCommand > { let mut queue = self . queue . lock () ; let count = queue . len () . min (max_count) ; let batch : Vec < _ > = queue . drain (.. count) . collect () ; self . has_work . store (! queue . is_empty () , Ordering :: Release) ; batch } pub fn is_empty (& self) -> bool { ! self . has_work . load (Ordering :: Acquire) } pub fn len (& self) -> usize { self . queue . lock () . len () } } . self_ty` (line 0, priv)

## File: MMSB/src/04_propagation/sparse_message_passing.rs

- Layer(s): 04_propagation
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `enqueue_sparse` (line 0, pub)
  - Signature: `# [doc = " Minimal placeholder for sparse message passing."] pub fn enqueue_sparse (queue : & PropagationQueue , comm...`
  - Calls: push

