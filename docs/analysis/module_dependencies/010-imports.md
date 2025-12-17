# Module Imports

## MMSB/src/00_physical/allocator_stats.rs (00_physical)

Module `allocator_stats`

- `std :: sync :: atomic :: { AtomicU64 , Ordering }`

## MMSB/src/00_physical/gpu_memory_pool.rs (00_physical)

Module `gpu_memory_pool`

- `parking_lot :: Mutex`
- `std :: collections :: HashMap`
- `std :: ffi :: c_void`

## MMSB/src/00_physical/nccl_integration.rs (00_physical)

Module `nccl_integration`

- `parking_lot :: Mutex`
- `std :: collections :: HashMap`
- `std :: ffi :: c_void`

## MMSB/src/01_page/allocator.rs (01_page)

Module `allocator`

- `crate :: page :: { Delta , DeltaID , Source }`
- `crate :: page :: { Page , Epoch }`
- `crate :: types :: { PageError , PageID , PageLocation }`
- `parking_lot :: Mutex`
- `std :: collections :: HashMap`
- `std :: ffi :: c_void`
- `std :: sync :: atomic :: AtomicU64`
- `super :: *`

## MMSB/src/01_page/checkpoint.rs (01_page)

Module `checkpoint`

- `crate :: page :: { PageAllocator , PageSnapshotData , PageID , PageLocation }`
- `std :: fs :: File`
- `std :: io :: { BufReader , BufWriter , Read , Write }`
- `std :: path :: Path`
- `super :: tlog :: TransactionLog`

## MMSB/src/01_page/delta.rs (01_page)

Module `delta`

- `crate :: types :: { DeltaError , DeltaID , Epoch , PageError , PageID , Source }`
- `std :: time :: { SystemTime , UNIX_EPOCH }`
- `super :: page :: Page`

## MMSB/src/01_page/delta_merge.rs (01_page)

Module `delta_merge`

- `# [cfg (target_arch = "x86_64")] use std :: arch :: x86_64 :: *`
- `crate :: page :: { Delta , DeltaError }`

## MMSB/src/01_page/delta_validation.rs (01_page)

Module `delta_validation`

- `crate :: page :: { Delta , DeltaError }`

## MMSB/src/01_page/device.rs (01_page)

Module `device`

- `crate :: page :: Page`
- `crate :: types :: PageID`
- `parking_lot :: RwLock`
- `std :: collections :: HashMap`
- `std :: sync :: Arc`

## MMSB/src/01_page/device_registry.rs (01_page)

Module `device_registry`

- `crate :: page :: { Page , PageID }`
- `parking_lot :: RwLock`
- `std :: collections :: HashMap`
- `std :: sync :: Arc`

## MMSB/src/01_page/host_device_sync.rs (01_page)

Module `host_device_sync`

- `crate :: types :: PageID`

## MMSB/src/01_page/lockfree_allocator.rs (01_page)

Module `lockfree_allocator`

- `crate :: page :: Page`
- `crate :: types :: { PageID , PageLocation }`
- `std :: ptr`
- `std :: sync :: atomic :: { AtomicPtr , AtomicU64 , Ordering }`

## MMSB/src/01_page/page.rs (01_page)

Module `page`

- `crate :: types :: { Epoch , EpochCell , PageError , PageID , PageLocation , DeltaError }`
- `parking_lot :: RwLock`
- `std :: convert :: TryInto`
- `std :: ffi :: c_void`
- `std :: ptr`
- `std :: sync :: Arc`
- `std :: sync :: atomic :: { AtomicU64 , Ordering }`
- `super :: delta :: Delta`
- `super :: delta_validation`

## MMSB/src/01_page/tlog.rs (01_page)

Module `tlog`

- `crate :: page :: { Delta , DeltaID , Epoch , PageID , Source }`
- `parking_lot :: RwLock`
- `std :: collections :: VecDeque`
- `std :: fs :: { File , OpenOptions }`
- `std :: io :: { BufReader , BufWriter , Read , Seek , SeekFrom , Write }`
- `std :: path :: { Path , PathBuf }`

## MMSB/src/01_page/tlog_compression.rs (01_page)

Module `tlog_compression`

- `crate :: page :: Delta`

## MMSB/src/01_page/tlog_replay.rs (01_page)

Module `tlog_replay`

- `crate :: page :: { Delta , Page }`

## MMSB/src/01_page/tlog_serialization.rs (01_page)

Module `tlog_serialization`

- `crate :: page :: { Delta , DeltaID , Epoch , PageID , Source }`
- `std :: fs :: File`
- `std :: io :: { BufReader , Read }`
- `std :: path :: Path`

## MMSB/src/01_types/delta_types.rs (01_types)

Module `delta_types`

- `super :: page_types :: PageID`
- `thiserror :: Error`

## MMSB/src/01_types/epoch_types.rs (01_types)

Module `epoch_types`

- `std :: sync :: atomic :: { AtomicU32 , Ordering }`

## MMSB/src/01_types/page_types.rs (01_types)

Module `page_types`

- `std :: fmt`
- `thiserror :: Error`

## MMSB/src/02_semiring/semiring_ops.rs (02_semiring)

Module `semiring_ops`

- `super :: semiring_types :: Semiring`

## MMSB/src/02_semiring/standard_semirings.rs (02_semiring)

Module `standard_semirings`

- `super :: semiring_types :: Semiring`

## MMSB/src/03_dag/cycle_detection.rs (03_dag)

Module `cycle_detection`

- `crate :: page :: PageID`
- `std :: collections :: HashMap`
- `super :: shadow_graph :: ShadowPageGraph`

## MMSB/src/03_dag/shadow_graph.rs (03_dag)

Module `shadow_graph`

- `crate :: page :: PageID`
- `parking_lot :: RwLock`
- `std :: collections :: { HashMap , HashSet }`
- `super :: edge_types :: EdgeType`

## MMSB/src/03_dag/shadow_graph_traversal.rs (03_dag)

Module `shadow_graph_traversal`

- `crate :: page :: PageID`
- `std :: collections :: { HashMap , VecDeque }`
- `super :: shadow_graph :: ShadowPageGraph`

## MMSB/src/04_propagation/propagation_command_buffer.rs (04_propagation)

Module `propagation_command_buffer`

- `crate :: page :: { Page , PageID }`
- `std :: sync :: Arc`

## MMSB/src/04_propagation/propagation_engine.rs (04_propagation)

Module `propagation_engine`

- `crate :: page :: { Page , PageID }`
- `parking_lot :: RwLock`
- `std :: collections :: HashMap`
- `std :: sync :: Arc`
- `super :: propagation_command_buffer :: PropagationCommand`
- `super :: propagation_queue :: PropagationQueue`

## MMSB/src/04_propagation/propagation_fastpath.rs (04_propagation)

Module `propagation_fastpath`

- `crate :: page :: Page`

## MMSB/src/04_propagation/propagation_queue.rs (04_propagation)

Module `propagation_queue`

- `parking_lot :: Mutex`
- `std :: collections :: VecDeque`
- `std :: sync :: Arc`
- `std :: sync :: atomic :: { AtomicBool , Ordering }`
- `super :: propagation_command_buffer :: PropagationCommand`

## MMSB/src/04_propagation/sparse_message_passing.rs (04_propagation)

Module `sparse_message_passing`

- `super :: propagation_command_buffer :: PropagationCommand`
- `super :: propagation_queue :: PropagationQueue`

## MMSB/src/05_adaptive/locality_optimizer.rs (05_adaptive)

Module `locality_optimizer`

- `std :: collections :: HashMap`
- `super :: *`

## MMSB/src/05_adaptive/memory_layout.rs (05_adaptive)

Module `memory_layout`

- `std :: collections :: HashMap`
- `super :: *`

## MMSB/src/05_adaptive/page_clustering.rs (05_adaptive)

Module `page_clustering`

- `std :: collections :: { HashMap , HashSet }`
- `super :: *`

## MMSB/src/06_utility/cpu_features.rs (06_utility)

Module `cpu_features`

- `std :: sync :: OnceLock`

## MMSB/src/06_utility/telemetry.rs (06_utility)

Module `telemetry`

- `std :: sync :: atomic :: { AtomicU64 , Ordering }`
- `std :: time :: Instant`
- `super :: *`

## MMSB/src/ffi.rs (root)

Module `ffi`

- `crate :: page :: checkpoint`
- `crate :: page :: tlog :: { TransactionLog , TransactionLogReader }`
- `crate :: page :: { Delta , DeltaID , Epoch , Page , PageAllocator , PageAllocatorConfig , PageError , PageID , PageLocation , Source }`
- `crate :: semiring :: { accumulate , fold_add , fold_mul , BooleanSemiring , Semiring , TropicalSemiring , }`
- `std :: cell :: RefCell`
- `std :: cmp :: min`
- `std :: ffi :: CStr`
- `std :: io :: ErrorKind`
- `std :: os :: raw :: c_char`
- `std :: ptr`
- `std :: slice`
- `std :: thread_local`

## MMSB/tests/delta_validation.rs (root)

Module `delta_validation`

- `mmsb_core :: page :: { validate_delta , Delta , DeltaID , Epoch , PageID , Source }`

## MMSB/tests/examples_basic.rs (root)

Module `examples_basic`

- `mmsb_core :: page :: { Delta , DeltaID , Epoch , Page , PageID , PageLocation , Source }`
- `mmsb_core :: page :: { write_checkpoint , load_checkpoint }`
- `mmsb_core :: physical :: { PageAllocator , PageAllocatorConfig }`

## MMSB/tests/mmsb_tests.rs (root)

Module `mmsb_tests`

- `mmsb_core :: page :: tlog :: TransactionLog`
- `mmsb_core :: page :: { Delta , DeltaID , Epoch , Page , PageID , Source }`
- `mmsb_core :: page :: { load_checkpoint , write_checkpoint }`
- `mmsb_core :: physical :: { PageAllocator , PageAllocatorConfig }`
- `std :: sync :: Arc`

## MMSB/tests/week27_31_integration.rs (root)

Module `week27_31_integration`

- `mmsb_core :: page :: merge_deltas`
- `mmsb_core :: page :: { Delta , DeltaID , Epoch , Page , PageID , PageLocation , Source }`
- `mmsb_core :: physical :: lockfree_allocator :: LockFreeAllocator`
- `mmsb_core :: physical :: { PageAllocator , PageAllocatorConfig }`
- `mmsb_core :: propagation :: PropagationQueue`
- `mmsb_core :: semiring :: TropicalSemiring`
- `mmsb_core :: utility :: CpuFeatures`
- `std :: sync :: Arc`

