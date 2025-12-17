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
- `crate :: physical :: AllocatorStats`
- `crate :: types :: { PageError , PageID , PageLocation }`
- `parking_lot :: Mutex`
- `std :: collections :: HashMap`
- `std :: ffi :: c_void`
- `std :: sync :: Arc`
- `std :: sync :: atomic :: AtomicU64`
- `super :: *`

## MMSB/src/01_page/checkpoint.rs (01_page)

Module `checkpoint`

- `crate :: page :: { PageAllocator , PageSnapshotData , PageID , PageLocation }`
- `std :: fs :: File`
- `std :: io :: { BufReader , BufWriter , Read , Write }`
- `std :: path :: Path`
- `super :: tlog :: TransactionLog`

## MMSB/src/01_page/columnar_delta.rs (01_page)

Module `columnar_delta`

- `crate :: page :: { Delta , DeltaID , Page , PageID , PageLocation , Source }`
- `crate :: types :: Epoch`
- `crate :: types :: { DeltaID , Epoch , PageError , PageID , Source }`
- `std :: collections :: HashMap`
- `super :: ColumnarDeltaBatch`
- `super :: delta :: Delta`
- `super :: page :: Page`

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

## MMSB/src/01_page/integrity_checker.rs (01_page)

Module `integrity_checker`

- `crate :: page :: { Delta , DeltaID , DeviceBufferRegistry , Page , PageID , PageLocation , Source }`
- `crate :: types :: Epoch`
- `crate :: types :: { DeltaID , Epoch , PageID }`
- `std :: collections :: HashMap`
- `std :: sync :: Arc`
- `super :: delta :: Delta`
- `super :: device_registry :: DeviceBufferRegistry`
- `super :: { DeltaIntegrityChecker , IntegrityViolationKind }`

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

## MMSB/src/01_page/replay_validator.rs (01_page)

Module `replay_validator`

- `crate :: page :: tlog :: TransactionLog`
- `crate :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation }`
- `crate :: types :: Epoch`
- `crate :: types :: { PageError , PageID }`
- `std :: collections :: HashMap`
- `std :: fs`
- `std :: path :: PathBuf`
- `std :: time :: { SystemTime , UNIX_EPOCH }`
- `super :: ReplayValidator`
- `super :: allocator :: { PageAllocator , PageSnapshotData }`
- `super :: tlog :: TransactionLog`

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

## MMSB/src/02_semiring/purity_validator.rs (02_semiring)

Module `purity_validator`

- `crate :: semiring :: { BooleanSemiring , TropicalSemiring }`
- `crate :: semiring :: { fold_add , fold_mul , Semiring }`
- `std :: fmt :: Debug`
- `std :: sync :: atomic :: { AtomicU32 , Ordering }`
- `super :: PurityValidator`

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

## MMSB/src/03_dag/graph_validator.rs (03_dag)

Module `graph_validator`

- `crate :: dag :: { EdgeType , ShadowPageGraph }`
- `crate :: types :: PageID`
- `std :: collections :: { HashMap , HashSet }`
- `std :: time :: { Duration , Instant }`
- `super :: GraphValidator`
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

- `crate :: page :: { Page , PageID , PageLocation }`
- `std :: sync :: Arc`
- `std :: sync :: atomic :: { AtomicBool , Ordering }`
- `std :: thread`
- `super :: *`
- `super :: propagation_command_buffer :: PropagationCommand`
- `super :: ring_buffer :: LockFreeRingBuffer`

## MMSB/src/04_propagation/ring_buffer.rs (04_propagation)

Module `ring_buffer`

- `std :: cell :: UnsafeCell`
- `std :: iter :: FromIterator`
- `std :: mem :: MaybeUninit`
- `std :: num :: NonZeroUsize`
- `std :: sync :: Arc`
- `std :: sync :: atomic :: { AtomicUsize , Ordering }`
- `std :: thread`
- `std :: time :: Duration`
- `super :: LockFreeRingBuffer`

## MMSB/src/04_propagation/sparse_message_passing.rs (04_propagation)

Module `sparse_message_passing`

- `super :: propagation_command_buffer :: PropagationCommand`
- `super :: propagation_queue :: PropagationQueue`

## MMSB/src/04_propagation/throughput_engine.rs (04_propagation)

Module `throughput_engine`

- `crate :: page :: { ColumnarDeltaBatch , Delta , PageAllocator , PageError , PageID , merge_deltas }`
- `crate :: page :: { DeltaID , PageAllocator , PageID , PageLocation , Source }`
- `crate :: types :: DeltaError`
- `crate :: types :: Epoch`
- `std :: collections :: HashMap`
- `std :: sync :: mpsc`
- `std :: sync :: { Arc , Mutex }`
- `std :: thread`
- `std :: time :: { Duration , Instant }`
- `super :: *`

## MMSB/src/04_propagation/tick_orchestrator.rs (04_propagation)

Module `tick_orchestrator`

- `crate :: dag :: { EdgeType , ShadowPageGraph }`
- `crate :: dag :: { GraphValidator , ShadowPageGraph }`
- `crate :: page :: { Delta , DeltaID , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source }`
- `crate :: page :: { Delta , PageError }`
- `crate :: types :: Epoch`
- `crate :: utility :: MemoryMonitor`
- `crate :: utility :: { MemoryMonitor , MemoryMonitorConfig }`
- `std :: sync :: Arc`
- `std :: time :: { Duration , Instant }`
- `super :: ThroughputEngine`
- `super :: TickOrchestrator`
- `super :: throughput_engine :: ThroughputEngine`

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

## MMSB/src/06_utility/invariant_checker.rs (06_utility)

Module `invariant_checker`

- `crate :: dag :: { EdgeType , ShadowPageGraph }`
- `crate :: dag :: { GraphValidator , ShadowPageGraph }`
- `crate :: page :: { DeviceBufferRegistry , PageAllocator , PageError , PageID }`
- `crate :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation }`
- `crate :: types :: Epoch`
- `parking_lot :: RwLock`
- `std :: collections :: HashMap`
- `super :: { EpochMonotonicity , GraphAcyclicity , InvariantChecker , InvariantContext }`

## MMSB/src/06_utility/memory_monitor.rs (06_utility)

Module `memory_monitor`

- `crate :: page :: PageAllocator`
- `crate :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation }`
- `crate :: physical :: AllocatorStats`
- `crate :: types :: Epoch`
- `crate :: types :: PageID`
- `parking_lot :: Mutex`
- `std :: collections :: { HashMap , HashSet }`
- `std :: sync :: Arc`
- `std :: time :: Duration`
- `std :: time :: { Duration , Instant }`
- `super :: { MemoryMonitor , MemoryMonitorConfig }`

## MMSB/src/06_utility/provenance_tracker.rs (06_utility)

Module `provenance_tracker`

- `crate :: dag :: ShadowPageGraph`
- `crate :: dag :: { EdgeType , ShadowPageGraph }`
- `crate :: page :: PageID`
- `std :: collections :: { HashMap , VecDeque }`
- `std :: sync :: Arc`
- `std :: time :: Instant`
- `super :: ProvenanceTracker`

## MMSB/src/06_utility/telemetry.rs (06_utility)

Module `telemetry`

- `std :: sync :: atomic :: { AtomicU64 , Ordering }`
- `std :: time :: Instant`
- `super :: *`

## MMSB/src/bin/phase6_bench.rs (root)

Module `phase6_bench`

- `mmsb_core :: dag :: { EdgeType , ShadowPageGraph }`
- `mmsb_core :: page :: { Delta , DeltaID , Epoch , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source , }`
- `mmsb_core :: propagation :: { ThroughputEngine , TickOrchestrator }`
- `mmsb_core :: utility :: { MemoryMonitor , MemoryMonitorConfig }`
- `std :: error :: Error`
- `std :: fs :: File`
- `std :: io :: Write`
- `std :: sync :: Arc`
- `std :: time :: { SystemTime , UNIX_EPOCH }`

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

## MMSB/tests/benchmark_01_replay.rs (root)

Module `benchmark_01_replay`

- `mmsb_core :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation , ReplayValidator , TransactionLog , }`
- `std :: path :: PathBuf`
- `std :: time :: { SystemTime , UNIX_EPOCH }`

## MMSB/tests/benchmark_02_integrity.rs (root)

Module `benchmark_02_integrity`

- `mmsb_core :: page :: { Delta , DeltaID , DeltaIntegrityChecker , DeviceBufferRegistry , Epoch , Page , PageID , PageLocation , Source , }`
- `std :: sync :: Arc`

## MMSB/tests/benchmark_03_graph.rs (root)

Module `benchmark_03_graph`

- `mmsb_core :: dag :: { EdgeType , GraphValidator , ShadowPageGraph }`
- `mmsb_core :: page :: PageID`

## MMSB/tests/benchmark_04_purity.rs (root)

Module `benchmark_04_purity`

- `mmsb_core :: semiring :: { BooleanSemiring , PurityValidator , TropicalSemiring }`

## MMSB/tests/benchmark_05_throughput.rs (root)

Module `benchmark_05_throughput`

- `mmsb_core :: page :: { Delta , DeltaID , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source }`
- `mmsb_core :: propagation :: ThroughputEngine`
- `mmsb_core :: types :: Epoch`
- `std :: sync :: Arc`

## MMSB/tests/benchmark_06_tick_latency.rs (root)

Module `benchmark_06_tick_latency`

- `mmsb_core :: dag :: { EdgeType , ShadowPageGraph }`
- `mmsb_core :: page :: { Delta , DeltaID , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source , }`
- `mmsb_core :: propagation :: { ThroughputEngine , TickOrchestrator }`
- `mmsb_core :: types :: Epoch`
- `mmsb_core :: utility :: { MemoryMonitor , MemoryMonitorConfig }`
- `std :: sync :: Arc`

## MMSB/tests/benchmark_07_memory.rs (root)

Module `benchmark_07_memory`

- `mmsb_core :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation }`
- `mmsb_core :: utility :: { MemoryMonitor , MemoryMonitorConfig }`
- `std :: sync :: Arc`

## MMSB/tests/benchmark_08_invariants.rs (root)

Module `benchmark_08_invariants`

- `mmsb_core :: dag :: ShadowPageGraph`
- `mmsb_core :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation }`
- `mmsb_core :: utility :: { InvariantChecker , InvariantContext }`
- `std :: sync :: Arc`

## MMSB/tests/benchmark_10_provenance.rs (root)

Module `benchmark_10_provenance`

- `mmsb_core :: dag :: { EdgeType , ShadowPageGraph }`
- `mmsb_core :: page :: PageID`
- `mmsb_core :: utility :: ProvenanceTracker`
- `std :: sync :: Arc`

## MMSB/tests/delta_validation.rs (root)

Module `delta_validation`

- `mmsb_core :: page :: { validate_delta , Delta , DeltaID , Epoch , PageID , Source }`

## MMSB/tests/examples_basic.rs (root)

Module `examples_basic`

- `mmsb_core :: page :: { Delta , DeltaID , Epoch , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source , }`

## MMSB/tests/mmsb_tests.rs (root)

Module `mmsb_tests`

- `mmsb_core :: page :: tlog :: TransactionLog`
- `mmsb_core :: page :: { load_checkpoint , write_checkpoint , Delta , DeltaID , Epoch , Page , PageAllocator , PageAllocatorConfig , PageID , Source , }`
- `std :: sync :: Arc`

## MMSB/tests/week27_31_integration.rs (root)

Module `week27_31_integration`

- `mmsb_core :: page :: LockFreeAllocator`
- `mmsb_core :: page :: merge_deltas`
- `mmsb_core :: page :: { Delta , DeltaID , Epoch , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source , }`
- `mmsb_core :: propagation :: PropagationQueue`
- `mmsb_core :: semiring :: TropicalSemiring`
- `mmsb_core :: utility :: CpuFeatures`
- `std :: sync :: Arc`

