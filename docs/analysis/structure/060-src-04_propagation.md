# Structure Group: src/04_propagation

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

