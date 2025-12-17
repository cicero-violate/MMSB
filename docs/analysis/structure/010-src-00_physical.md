# Structure Group: src/00_physical

## File: MMSB/src/00_physical/allocator_stats.rs

- Layer(s): 00_physical
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `AllocatorStats` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct AllocatorStats { allocations : AtomicU64 , frees : AtomicU64 , }`
- [Rust | Impl] `impl AllocatorStats { pub fn record_alloc (& self) { self . allocations . fetch_add (1 , Ordering :: Relaxed) ; } pub fn record_free (& self) { self . frees . fetch_add (1 , Ordering :: Relaxed) ; } pub fn snapshot (& self) -> (u64 , u64) { (self . allocations . load (Ordering :: Relaxed) , self . frees . load (Ordering :: Relaxed) ,) } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/gpu_memory_pool.rs

- Layer(s): 00_physical
- Language coverage: Rust (5)
- Element types: Impl (2), Struct (3)
- Total elements: 5

### Elements

- [Rust | Impl] `Drop for impl Drop for GPUMemoryPool { fn drop (& mut self) { self . clear () ; } } . self_ty` (line 0, priv)
- [Rust | Struct] `GPUMemoryPool` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct GPUMemoryPool { slabs : Mutex < HashMap < usize , Slab > > , stats : Mutex < PoolStats ...`
- [Rust | Struct] `PoolStats` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Default)] pub struct PoolStats { pub allocations : u64 , pub deallocations : u64 , pub cac...`
- [Rust | Struct] `Slab` (line 0, priv)
  - Signature: `# [derive (Debug)] struct Slab { size : usize , free_blocks : Vec < * mut c_void > , allocated_count : usize , }`
- [Rust | Impl] `impl GPUMemoryPool { pub fn new () -> Self { let mut slabs = HashMap :: new () ; for & size in SLAB_SIZES { slabs . insert (size , Slab { size , free_blocks : Vec :: new () , allocated_count : 0 , }) ; } Self { slabs : Mutex :: new (slabs) , stats : Mutex :: new (PoolStats :: default ()) , } } fn select_slab_size (& self , size : usize) -> usize { SLAB_SIZES . iter () . find (| & & s | s >= size) . copied () . unwrap_or (* SLAB_SIZES . last () . unwrap ()) } pub fn allocate (& self , size : usize) -> Result < * mut c_void , i32 > { let slab_size = self . select_slab_size (size) ; let mut slabs = self . slabs . lock () ; let mut stats = self . stats . lock () ; stats . allocations += 1 ; if let Some (slab) = slabs . get_mut (& slab_size) { if let Some (ptr) = slab . free_blocks . pop () { stats . cache_hits += 1 ; slab . allocated_count += 1 ; return Ok (ptr) ; } } stats . cache_misses += 1 ; stats . bytes_allocated += slab_size as u64 ; let mut ptr : * mut c_void = std :: ptr :: null_mut () ; let result = unsafe { cudaMalloc (& mut ptr , slab_size) } ; if result == 0 { if let Some (slab) = slabs . get_mut (& slab_size) { slab . allocated_count += 1 ; } Ok (ptr) } else { Err (result) } } pub fn deallocate (& self , ptr : * mut c_void , size : usize) { let slab_size = self . select_slab_size (size) ; let mut slabs = self . slabs . lock () ; let mut stats = self . stats . lock () ; stats . deallocations += 1 ; if let Some (slab) = slabs . get_mut (& slab_size) { slab . free_blocks . push (ptr) ; slab . allocated_count -= 1 ; stats . bytes_cached += slab_size as u64 ; } } pub fn get_stats (& self) -> PoolStats { self . stats . lock () . clone () } pub fn clear (& self) { let mut slabs = self . slabs . lock () ; for slab in slabs . values_mut () { for ptr in slab . free_blocks . drain (..) { unsafe { cudaFree (ptr) } ; } } let mut stats = self . stats . lock () ; stats . bytes_cached = 0 ; } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/mod.rs

- Layer(s): 00_physical
- Language coverage: Rust (3)
- Element types: Module (3)
- Total elements: 3

### Elements

- [Rust | Module] `allocator_stats` (line 0, pub)
- [Rust | Module] `gpu_memory_pool` (line 0, pub)
- [Rust | Module] `nccl_integration` (line 0, pub)

## File: MMSB/src/00_physical/nccl_integration.rs

- Layer(s): 00_physical
- Language coverage: Rust (6)
- Element types: Enum (2), Impl (2), Struct (2)
- Total elements: 6

### Elements

- [Rust | Impl] `Drop for impl Drop for NCCLContext { fn drop (& mut self) { let comms = self . communicators . lock () ; for comm in comms . values () { unsafe { ncclCommDestroy (comm . comm) } ; } } } . self_ty` (line 0, priv)
- [Rust | Struct] `NCCLCommunicator` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct NCCLCommunicator { comm : ncclComm_t , rank : i32 , world_size : i32 , }`
- [Rust | Struct] `NCCLContext` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct NCCLContext { communicators : Mutex < HashMap < i32 , NCCLCommunicator > > , unique_id ...`
- [Rust | Impl] `impl NCCLContext { pub fn new (_num_gpus : i32) -> Result < Self , i32 > { let mut unique_id = [0u8 ; 128] ; let result = unsafe { ncclGetUniqueId (& mut unique_id) } ; if result != 0 { return Err (result) ; } Ok (Self { communicators : Mutex :: new (HashMap :: new ()) , unique_id , }) } pub fn init_communicator (& self , rank : i32 , world_size : i32) -> Result < () , i32 > { let mut comm : ncclComm_t = std :: ptr :: null_mut () ; let result = unsafe { ncclCommInitRank (& mut comm , world_size , self . unique_id , rank) } ; if result != 0 { return Err (result) ; } let communicator = NCCLCommunicator { comm , rank , world_size , } ; self . communicators . lock () . insert (rank , communicator) ; Ok (()) } pub fn all_reduce (& self , rank : i32 , sendbuf : * const c_void , recvbuf : * mut c_void , count : usize , datatype : ncclDataType_t , op : ncclRedOp_t , stream : * mut c_void ,) -> Result < () , i32 > { let comms = self . communicators . lock () ; let comm = comms . get (& rank) . ok_or (- 1) ? ; let result = unsafe { ncclAllReduce (sendbuf , recvbuf , count , datatype , op , comm . comm , stream) } ; if result == 0 { Ok (()) } else { Err (result) } } pub fn all_gather (& self , rank : i32 , sendbuf : * const c_void , recvbuf : * mut c_void , sendcount : usize , datatype : ncclDataType_t , stream : * mut c_void ,) -> Result < () , i32 > { let comms = self . communicators . lock () ; let comm = comms . get (& rank) . ok_or (- 1) ? ; let result = unsafe { ncclAllGather (sendbuf , recvbuf , sendcount , datatype , comm . comm , stream) } ; if result == 0 { Ok (()) } else { Err (result) } } } . self_ty` (line 0, priv)
- [Rust | Enum] `ncclDataType_t` (line 0, pub)
- [Rust | Enum] `ncclRedOp_t` (line 0, pub)

