# Structure Group: src/00_physical

## File: MMSB/src/00_physical/DeviceFallback.jl

- Layer(s): 00_physical
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `DeviceFallback` (line 1, pub)
- [Julia | Function] `has_gpu_support` (line 5, pub)
  - Signature: `has_gpu_support()`
  - Calls: CUDA.functional
- [Julia | Struct] `CPUPropagationQueue` (line 6, pub)
  - Signature: `struct CPUPropagationQueue`
- [Julia | Function] `CPUPropagationQueue` (line 9, pub)
  - Signature: `CPUPropagationQueue()`
  - Calls: CPUPropagationQueue
- [Julia | Function] `fallback_to_cpu` (line 10, pub)
  - Signature: `fallback_to_cpu(f)`
  - Calls: f

## File: MMSB/src/00_physical/DeviceSync.jl

- Layer(s): 00_physical
- Language coverage: Julia (14)
- Element types: Function (12), Module (1), Struct (1)
- Total elements: 14

### Elements

- [Julia | Module] `DeviceSync` (line 8, pub)
- [Julia | Struct] `GPUCommandBuffer` (line 21, pub)
  - Signature: `mutable struct GPUCommandBuffer`
- [Julia | Function] `create_gpu_command_buffer` (line 38, pub)
  - Signature: `create_gpu_command_buffer(capacity::UInt32`
- [Julia | Function] `enqueue_propagation_command` (line 60, pub)
  - Signature: `enqueue_propagation_command(buf::GPUCommandBuffer, page::Page, deps::Vector{Page})`
- [Julia | Function] `wait_gpu_queue` (line 72, pub)
  - Signature: `wait_gpu_queue(buf::GPUCommandBuffer)`
  - Calls: ccall, time_ns
- [Julia | Function] `sync_page_to_gpu!` (line 99, pub)
  - Signature: `sync_page_to_gpu!(page::Page)`
  - Calls: CuArray
- [Julia | Function] `sync_page_to_cpu!` (line 130, pub)
  - Signature: `sync_page_to_cpu!(page::Page)`
  - Calls: CUDA.synchronize, CUDA.unsafe_free!
- [Julia | Function] `sync_bidirectional!` (line 164, pub)
  - Signature: `sync_bidirectional!(page1::Page, page2::Page)`
  - Calls: CPU, sync_page_to_gpu!
- [Julia | Function] `ensure_page_on_device!` (line 194, pub)
  - Signature: `ensure_page_on_device!(page::Page, target::PageLocation)::Bool`
  - Calls: UnsupportedLocationError, string, sync_page_to_cpu!, sync_page_to_gpu!, throw
- [Julia | Function] `async_sync_page_to_gpu!` (line 222, pub)
  - Signature: `async_sync_page_to_gpu!(page::Page, stream::CuStream)`
  - Calls: CuArray
- [Julia | Function] `batch_sync_to_gpu!` (line 253, pub)
  - Signature: `batch_sync_to_gpu!(pages::Vector{Page})`
  - Calls: filter, isempty, sync_page_to_gpu!
- [Julia | Function] `batch_sync_to_cpu!` (line 280, pub)
  - Signature: `batch_sync_to_cpu!(pages::Vector{Page})`
  - Calls: CUDA.synchronize, filter, isempty, sync_page_to_cpu!
- [Julia | Function] `get_sync_statistics` (line 309, pub)
  - Signature: `get_sync_statistics(state::MMSBState)::Dict{Symbol, Any}`
- [Julia | Function] `prefetch_pages_to_gpu!` (line 351, pub)
  - Signature: `prefetch_pages_to_gpu!(state::MMSBState, page_ids::Vector{PageID})`
  - Calls: batch_sync_to_gpu!, filter, get_page

## File: MMSB/src/00_physical/GPUKernels.jl

- Layer(s): 00_physical
- Language coverage: Julia (12)
- Element types: Function (11), Module (1)
- Total elements: 12

### Elements

- [Julia | Module] `GPUKernels` (line 8, pub)
- [Julia | Function] `delta_merge_kernel!` (line 33, pub)
  - Signature: `delta_merge_kernel!(base::CuDeviceArray{UInt8,1}, mask::CuDeviceArray{Bool,1}, delta::CuDeviceArray{UInt8,1})`
  - Calls: blockDim, blockIdx, length, threadIdx
- [Julia | Function] `launch_delta_merge!` (line 62, pub)
  - Signature: `launch_delta_merge!(base::CuArray{UInt8}, mask::CuArray{Bool}, delta::CuArray{UInt8})`
  - Calls: compute_optimal_kernel_config, delta_merge_kernel!, length
- [Julia | Function] `page_copy_kernel!` (line 83, pub)
  - Signature: `page_copy_kernel!(dest::CuDeviceArray{UInt8,1}, src::CuDeviceArray{UInt8,1}, n::Int)`
  - Calls: blockDim, blockIdx, threadIdx
- [Julia | Function] `launch_page_copy!` (line 100, pub)
  - Signature: `launch_page_copy!(dest::CuArray{UInt8}, src::CuArray{UInt8}, n::Int)`
  - Calls: compute_optimal_kernel_config, page_copy_kernel!
- [Julia | Function] `page_zero_kernel!` (line 113, pub)
  - Signature: `page_zero_kernel!(data::CuDeviceArray{UInt8,1})`
  - Calls: blockDim, blockIdx, length, threadIdx
- [Julia | Function] `launch_page_zero!` (line 126, pub)
  - Signature: `launch_page_zero!(data::CuArray{UInt8})`
  - Calls: compute_optimal_kernel_config, length, page_zero_kernel!
- [Julia | Function] `page_compare_kernel!` (line 143, pub)
  - Signature: `page_compare_kernel!(result::CuDeviceArray{Bool,1}, page1::CuDeviceArray{UInt8,1}, page2::CuDeviceArray{UInt8,1})`
  - Calls: blockDim, blockIdx, length, threadIdx
- [Julia | Function] `launch_page_compare!` (line 160, pub)
  - Signature: `launch_page_compare!(result::CuArray{Bool}, page1::CuArray{UInt8}, page2::CuArray{UInt8})`
  - Calls: compute_optimal_kernel_config, length, page_compare_kernel!
- [Julia | Function] `sparse_delta_apply_kernel!` (line 187, pub)
  - Signature: `sparse_delta_apply_kernel!(base::CuDeviceArray{UInt8,1}, indices::CuDeviceArray{Int32,1}, values::CuDeviceArray{UInt8...`
  - Calls: blockDim, blockIdx, threadIdx
- [Julia | Function] `launch_sparse_delta_apply!` (line 206, pub)
  - Signature: `launch_sparse_delta_apply!(base::CuArray{UInt8}, indices::CuArray{Int32}, values::CuArray{UInt8})`
  - Calls: compute_optimal_kernel_config, length, sparse_delta_apply_kernel!
- [Julia | Function] `compute_optimal_kernel_config` (line 230, pub)
  - Signature: `compute_optimal_kernel_config(data_size::Int)::Tuple{Int, Int}`

## File: MMSB/src/00_physical/PageAllocator.jl

- Layer(s): 00_physical
- Language coverage: Julia (7)
- Element types: Function (6), Module (1)
- Total elements: 7

### Elements

- [Julia | Module] `PageAllocator` (line 8, pub)
- [Julia | Function] `create_page!` (line 22, pub)
  - Signature: `create_page!(state::MMSBState, size::Int64, location::PageLocation; metadata::Dict{Symbol,Any}`
- [Julia | Function] `delete_page!` (line 49, pub)
  - Signature: `delete_page!(state::MMSBState, page_id::PageID)`
  - Calls: PageNotFoundError, UInt64, delete!, get, get_children, get_parents, lock, remove_dependency!, throw
- [Julia | Function] `migrate_page!` (line 72, pub)
  - Signature: `migrate_page!(state::MMSBState, page_id::PageID, target_location::PageLocation)::Page`
  - Calls: PageNotFoundError, UInt64, get, lock, throw
- [Julia | Function] `resize_page!` (line 86, pub)
  - Signature: `resize_page!(state::MMSBState, page_id::PageID, new_size::Int64)::Page`
  - Calls: PageNotFoundError, UInt64, get, lock, throw
- [Julia | Function] `allocate_page_arrays` (line 99, pub)
  - Signature: `allocate_page_arrays(size::Int64, location::PageLocation)`
  - Calls: error
- [Julia | Function] `clone_page` (line 106, pub)
  - Signature: `clone_page(state::MMSBState, page_id::PageID)::Page`
  - Calls: error

## File: MMSB/src/00_physical/UnifiedMemory.jl

- Layer(s): 00_physical
- Language coverage: Julia (15)
- Element types: Function (13), Module (1), Struct (1)
- Total elements: 15

### Elements

- [Julia | Module] `UnifiedMemory` (line 8, pub)
- [Julia | Struct] `GPUMemoryPool` (line 22, pub)
  - Signature: `mutable struct GPUMemoryPool`
- [Julia | Function] `GPUMemoryPool` (line 39, pub)
  - Signature: `GPUMemoryPool()::GPUMemoryPool`
  - Calls: GPUMemoryPool, ccall
- [Julia | Function] `allocate_from_pool` (line 54, pub)
  - Signature: `allocate_from_pool(pool::GPUMemoryPool, size::UInt64)::Ptr{Cvoid}`
  - Calls: ccall
- [Julia | Function] `deallocate_to_pool` (line 66, pub)
  - Signature: `deallocate_to_pool(pool::GPUMemoryPool, ptr::Ptr{Cvoid}, size::UInt64)`
  - Calls: ccall
- [Julia | Function] `get_pool_stats` (line 76, pub)
  - Signature: `get_pool_stats(pool::GPUMemoryPool)::Dict{Symbol, UInt64}`
  - Calls: ccall
- [Julia | Function] `is_unified_memory_available` (line 99, pub)
  - Signature: `is_unified_memory_available()::Bool`
  - Calls: CUDA.capability, CUDA.device, CUDA.functional
- [Julia | Function] `create_unified_page!` (line 128, pub)
  - Signature: `create_unified_page!(state::MMSBState, size::Int64)::Page`
  - Calls: Page, allocate_page_id!, ccall
- [Julia | Function] `prefetch_unified_to_gpu!` (line 146, pub)
  - Signature: `prefetch_unified_to_gpu!(page::Page, device::CuDevice`
  - Calls: CUDA.device
- [Julia | Function] `prefetch_unified_to_cpu!` (line 168, pub)
  - Signature: `prefetch_unified_to_cpu!(page::Page)`
  - Calls: CUDA.functional, ccall, pointer, sizeof
- [Julia | Function] `adaptive_prefetch_distance` (line 194, pub)
  - Signature: `adaptive_prefetch_distance(latency_history::Vector{Float64})::Int`
  - Calls: isempty, mean
- [Julia | Function] `set_preferred_location!` (line 225, pub)
  - Signature: `set_preferred_location!(page::Page, device::Union{CuDevice, Symbol})`
  - Calls: CUDA.Mem.advise
- [Julia | Function] `convert_to_unified!` (line 249, pub)
  - Signature: `convert_to_unified!(page::Page)`
  - Calls: sync_page_to_cpu!
- [Julia | Function] `enable_read_mostly_hint!` (line 278, pub)
  - Signature: `enable_read_mostly_hint!(page::Page)`
  - Calls: CUDA.Mem.advise
- [Julia | Function] `disable_read_mostly_hint!` (line 292, pub)
  - Signature: `disable_read_mostly_hint!(page::Page)`
  - Calls: CUDA.Mem.advise

## File: MMSB/src/00_physical/allocator.rs

- Layer(s): 00_physical
- Language coverage: Rust (10)
- Element types: Function (3), Impl (2), Module (1), Struct (4)
- Total elements: 10

### Elements

- [Rust | Impl] `Default for impl Default for PageAllocatorConfig { fn default () -> Self { Self { default_location : PageLocation :: Cpu , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `PageAllocator` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct PageAllocator { config : PageAllocatorConfig , pages : Mutex < HashMap < PageID , Box <...`
- [Rust | Struct] `PageAllocatorConfig` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageAllocatorConfig { pub default_location : PageLocation , }`
- [Rust | Struct] `PageInfo` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageInfo { pub page_id : PageID , pub size : usize , pub location : PageLocatio...`
- [Rust | Struct] `PageSnapshotData` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageSnapshotData { pub page_id : PageID , pub size : usize , pub location : Pag...`
- [Rust | Impl] `impl PageAllocator { pub fn new (config : PageAllocatorConfig) -> Self { println ! ("Allocating new PageAllocator instance with config: {:?}" , config) ; Self { config , pages : Mutex :: new (HashMap :: new ()) , next_id : AtomicU64 :: new (1) , } } pub fn allocate_raw (& self , page_id_hint : PageID , size : usize , location : Option < PageLocation >) -> Result < * mut Page , PageError > { let loc = location . unwrap_or (self . config . default_location) ; if self . pages . lock () . contains_key (& page_id_hint) { return Err (PageError :: AlreadyExists (page_id_hint)) ; } let page = Box :: new (Page :: new (page_id_hint , size , loc) ?) ; let ptr = Box :: into_raw (page) ; println ! ("[ALLOCATOR] allocate_raw(id={}) → raw ptr = {:p}" , page_id_hint . 0 , ptr) ; self . pages . lock () . insert (page_id_hint , unsafe { Box :: from_raw (ptr) }) ; Ok (ptr) } pub fn free (& self , page_id : PageID) { if let Some (_) = self . pages . lock () . remove (& page_id) { println ! ("[ALLOCATOR] Freed page {}" , page_id . 0) ; } } pub fn release (& self , page_id : PageID) { if let Some (boxed_page) = self . pages . lock () . remove (& page_id) { println ! ("[ALLOCATOR] release({}): ownership transferred — Box removed from map but NOT dropped (caller now owns it)" , page_id . 0) ; std :: mem :: forget (boxed_page) ; } else { println ! ("[ALLOCATOR] release({}): page not found — already released?" , page_id . 0) ; } } pub fn acquire_page (& self , page_id : PageID) -> Option < * mut Page > { self . pages . lock () . get (& page_id) . map (| b | & * * b as * const Page as * mut Page) } pub fn len (& self) -> usize { self . pages . lock () . len () } pub fn page_infos (& self) -> Vec < PageInfo > { let pages = self . pages . lock () ; pages . values () . map (| page | PageInfo { page_id : page . id , size : page . size () , location : page . location () , epoch : page . epoch () . 0 , metadata : page . metadata_blob () , }) . collect () } pub fn snapshot_pages (& self) -> Vec < PageSnapshotData > { let pages = self . pages . lock () ; pages . values () . map (| page | PageSnapshotData { page_id : page . id , size : page . size () , location : page . location () , epoch : page . epoch () . 0 , metadata_blob : page . metadata_blob () , data : page . data_slice () . to_vec () , }) . collect () } pub fn restore_from_snapshot (& self , snapshots : Vec < PageSnapshotData >) -> Result < () , PageError > { eprintln ! ("\n=== RESTORE_FROM_SNAPSHOT STARTED ===") ; eprintln ! ("   Clearing {} existing pages" , self . pages . lock () . len ()) ; let mut pages = self . pages . lock () ; pages . clear () ; for (i , snapshot) in snapshots . iter () . enumerate () { eprintln ! ("   [{i}] Restoring page ID={:?} size={} epoch={} loc={:?}" , snapshot . page_id , snapshot . size , snapshot . epoch , snapshot . location) ; let mut page = match Page :: new (snapshot . page_id , snapshot . size , snapshot . location) { Ok (p) => Box :: new (p) , Err (e) => { eprintln ! ("      Page::new() FAILED: {e}") ; return Err (e) ; } } ; page . set_epoch (Epoch (snapshot . epoch)) ; eprintln ! ("      Epoch set to {}" , snapshot . epoch) ; let dst = page . data_mut_slice () ; if dst . len () != snapshot . data . len () { eprintln ! ("      FATAL: data size mismatch! page={} snapshot={}" , dst . len () , snapshot . data . len ()) ; return Err (PageError :: MetadataDecode ("data size mismatch in snapshot")) ; } dst . copy_from_slice (& snapshot . data) ; eprintln ! ("      Data copied ({} bytes)" , snapshot . data . len ()) ; eprintln ! ("      Applying metadata ({} bytes)..." , snapshot . metadata_blob . len ()) ; if let Err (e) = page . set_metadata_blob (& snapshot . metadata_blob) { eprintln ! ("      METADATA RESTORE FAILED: {e}") ; return Err (e) ; } eprintln ! ("      Metadata restored OK") ; pages . insert (snapshot . page_id , page) ; eprintln ! ("      Page inserted") ; } eprintln ! ("=== RESTORE_FROM_SNAPSHOT SUCCESS: {} pages restored ===" , snapshots . len ()) ; Ok (()) } } . self_ty` (line 0, priv)
- [Rust | Function] `test_checkpoint_roundtrip_in_memory` (line 0, priv)
  - Signature: `# [test] fn test_checkpoint_roundtrip_in_memory () { let alloc = PageAllocator :: new (PageAllocatorConfig :: default...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, apply_delta, DeltaID, PageID, Epoch, Source, into, snapshot_pages, expect, restore_from_snapshot, unwrap, acquire_page, PageID
- [Rust | Function] `test_page_info_metadata_roundtrip` (line 0, priv)
  - Signature: `# [test] fn test_page_info_metadata_roundtrip () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defau...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, set_metadata, page_infos
- [Rust | Function] `test_unified_page` (line 0, priv)
  - Signature: `# [test] fn test_unified_page () { let config = PageAllocatorConfig { default_location : PageLocation :: Unified , } ...`
  - Calls: PageAllocator::new, expect, allocate_raw, PageID, data_mut_slice
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/00_physical/allocator_stats.rs

- Layer(s): 00_physical
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `AllocatorStats` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct AllocatorStats { allocations : AtomicU64 , frees : AtomicU64 , }`
- [Rust | Impl] `impl AllocatorStats { pub fn record_alloc (& self) { self . allocations . fetch_add (1 , Ordering :: Relaxed) ; } pub fn record_free (& self) { self . frees . fetch_add (1 , Ordering :: Relaxed) ; } pub fn snapshot (& self) -> (u64 , u64) { (self . allocations . load (Ordering :: Relaxed) , self . frees . load (Ordering :: Relaxed) ,) } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/device.rs

- Layer(s): 00_physical
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `DeviceRegistry` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct DeviceRegistry { pages : RwLock < HashMap < PageID , Arc < Page > > > , }`
- [Rust | Impl] `impl DeviceRegistry { pub fn register (& self , page : Arc < Page >) { self . pages . write () . insert (page . id , page) ; } pub fn unregister (& self , page_id : PageID) { self . pages . write () . remove (& page_id) ; } pub fn get (& self , page_id : PageID) -> Option < Arc < Page > > { self . pages . read () . get (& page_id) . cloned () } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/device_registry.rs

- Layer(s): 00_physical
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `DeviceBufferRegistry` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct DeviceBufferRegistry { map : RwLock < HashMap < PageID , Arc < Page > > > , }`
- [Rust | Impl] `impl DeviceBufferRegistry { pub fn insert (& self , page : Arc < Page >) { self . map . write () . insert (page . id , page) ; } pub fn remove (& self , page_id : PageID) { self . map . write () . remove (& page_id) ; } pub fn len (& self) -> usize { self . map . read () . len () } } . self_ty` (line 0, priv)

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

## File: MMSB/src/00_physical/host_device_sync.rs

- Layer(s): 00_physical
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `HostDeviceSync` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct HostDeviceSync { pending : Vec < PageID > , }`
- [Rust | Impl] `impl HostDeviceSync { pub fn enqueue (& mut self , page_id : PageID) { self . pending . push (page_id) ; } pub fn drain (& mut self) -> Vec < PageID > { std :: mem :: take (& mut self . pending) } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/lockfree_allocator.rs

- Layer(s): 00_physical
- Language coverage: Rust (6)
- Element types: Impl (4), Struct (2)
- Total elements: 6

### Elements

- [Rust | Impl] `Drop for impl Drop for LockFreeAllocator { fn drop (& mut self) { self . clear () ; } } . self_ty` (line 0, priv)
- [Rust | Struct] `FreeListNode` (line 0, priv)
  - Signature: `struct FreeListNode { next : AtomicPtr < FreeListNode > , page_ptr : * mut Page , }`
- [Rust | Struct] `LockFreeAllocator` (line 0, pub)
  - Signature: `pub struct LockFreeAllocator { freelist_head : AtomicPtr < FreeListNode > , freelist_size : AtomicU64 , allocated_cou...`
- [Rust | Impl] `Send for unsafe impl Send for LockFreeAllocator { } . self_ty` (line 0, priv)
- [Rust | Impl] `Sync for unsafe impl Sync for LockFreeAllocator { } . self_ty` (line 0, priv)
- [Rust | Impl] `impl LockFreeAllocator { pub fn new () -> Self { Self { freelist_head : AtomicPtr :: new (ptr :: null_mut ()) , freelist_size : AtomicU64 :: new (0) , allocated_count : AtomicU64 :: new (0) , freed_count : AtomicU64 :: new (0) , } } pub fn try_allocate_small (& self , _page_id : PageID , size : usize , _location : PageLocation) -> Option < * mut Page > { if size > SMALL_PAGE_THRESHOLD { return None ; } loop { let head = self . freelist_head . load (Ordering :: Acquire) ; if head . is_null () { return None ; } let node = unsafe { & * head } ; let next = node . next . load (Ordering :: Relaxed) ; if self . freelist_head . compare_exchange (head , next , Ordering :: Release , Ordering :: Acquire) . is_ok () { self . freelist_size . fetch_sub (1 , Ordering :: Relaxed) ; self . allocated_count . fetch_add (1 , Ordering :: Relaxed) ; let page_ptr = node . page_ptr ; unsafe { Box :: from_raw (head) ; } return Some (page_ptr) ; } } } pub fn deallocate_small (& self , page_ptr : * mut Page) -> bool { let page = unsafe { & * page_ptr } ; if page . size () > SMALL_PAGE_THRESHOLD { return false ; } let current_size = self . freelist_size . load (Ordering :: Relaxed) ; if current_size >= FREELIST_CAPACITY as u64 { return false ; } let node = Box :: into_raw (Box :: new (FreeListNode { next : AtomicPtr :: new (ptr :: null_mut ()) , page_ptr , })) ; loop { let head = self . freelist_head . load (Ordering :: Acquire) ; unsafe { (* node) . next . store (head , Ordering :: Relaxed) ; } if self . freelist_head . compare_exchange (head , node , Ordering :: Release , Ordering :: Acquire) . is_ok () { self . freelist_size . fetch_add (1 , Ordering :: Relaxed) ; self . freed_count . fetch_add (1 , Ordering :: Relaxed) ; return true ; } } } pub fn get_stats (& self) -> (u64 , u64 , u64) { (self . freelist_size . load (Ordering :: Relaxed) , self . allocated_count . load (Ordering :: Relaxed) , self . freed_count . load (Ordering :: Relaxed) ,) } pub fn clear (& self) { let mut head = self . freelist_head . swap (ptr :: null_mut () , Ordering :: AcqRel) ; while ! head . is_null () { let node = unsafe { Box :: from_raw (head) } ; head = node . next . load (Ordering :: Relaxed) ; unsafe { drop (Box :: from_raw (node . page_ptr)) ; } } self . freelist_size . store (0 , Ordering :: Relaxed) ; } } . self_ty` (line 0, priv)

## File: MMSB/src/00_physical/mod.rs

- Layer(s): 00_physical
- Language coverage: Rust (8)
- Element types: Module (8)
- Total elements: 8

### Elements

- [Rust | Module] `allocator` (line 0, pub)
- [Rust | Module] `allocator_stats` (line 0, pub)
- [Rust | Module] `device` (line 0, pub)
- [Rust | Module] `device_registry` (line 0, pub)
- [Rust | Module] `gpu_memory_pool` (line 0, pub)
- [Rust | Module] `host_device_sync` (line 0, pub)
- [Rust | Module] `lockfree_allocator` (line 0, pub)
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

