# Module Exports

## MMSB/src/00_physical/mod.rs (00_physical)

Module `mod`

- `allocator :: { PageAllocator , PageAllocatorConfig , PageInfo , PageSnapshotData }`
- `allocator_stats :: AllocatorStats`
- `device :: DeviceRegistry`
- `device_registry :: DeviceBufferRegistry`
- `gpu_memory_pool :: { GPUMemoryPool , PoolStats }`
- `host_device_sync :: HostDeviceSync`
- `lockfree_allocator :: LockFreeAllocator`
- `nccl_integration :: { NCCLContext , ncclRedOp_t , ncclDataType_t }`

## MMSB/src/01_page/mod.rs (01_page)

Module `mod`

- `checkpoint :: { load_checkpoint , write_checkpoint }`
- `delta :: { Delta , DeltaError , DeltaID , Source }`
- `delta_merge :: merge_deltas`
- `delta_validation :: validate_delta`
- `epoch :: { Epoch , EpochCell }`
- `page :: { Metadata , Page , PageError , PageID , PageLocation }`
- `simd_mask :: generate_mask`
- `tlog :: { LogSummary , TransactionLog , TransactionLogReader , summary }`

## MMSB/src/01_types/mod.rs (01_types)

Module `mod`

- `crate :: page :: *`

## MMSB/src/02_semiring/mod.rs (02_semiring)

Module `mod`

- `semiring_ops :: { accumulate , fold_add , fold_mul }`
- `semiring_types :: Semiring`
- `standard_semirings :: { BooleanSemiring , TropicalSemiring }`

## MMSB/src/03_dag/mod.rs (03_dag)

Module `mod`

- `cycle_detection :: has_cycle`
- `edge_types :: EdgeType`
- `shadow_graph :: { Edge , ShadowPageGraph }`
- `shadow_graph_traversal :: topological_sort`

## MMSB/src/03_dag/shadow_graph_mod.rs (03_dag)

Module `shadow_graph_mod`

- `super :: edge_types :: EdgeType`
- `super :: shadow_graph :: { Edge , ShadowPageGraph }`
- `super :: shadow_graph_traversal :: topological_sort`

## MMSB/src/03_device/mod.rs (03_device)

Module `mod`

- `device :: DeviceRegistry as DeviceRegistryHandle`

## MMSB/src/04_propagation/mod.rs (04_propagation)

Module `mod`

- `propagation_command_buffer :: PropagationCommand`
- `propagation_engine :: PropagationEngine`
- `propagation_fastpath :: passthrough`
- `propagation_queue :: PropagationQueue`
- `sparse_message_passing :: enqueue_sparse`

## MMSB/src/05_adaptive/mod.rs (05_adaptive)

Module `mod`

- `locality_optimizer :: LocalityOptimizer`
- `memory_layout :: { MemoryLayout , AccessPattern , PageId , PhysAddr }`
- `page_clustering :: { PageCluster , PageClusterer }`

## MMSB/src/06_utility/mod.rs (06_utility)

Module `mod`

- `cpu_features :: CpuFeatures`
- `telemetry :: { Telemetry , TelemetrySnapshot }`

## MMSB/src/lib.rs (root)

Module `lib`

- `ffi :: *`

