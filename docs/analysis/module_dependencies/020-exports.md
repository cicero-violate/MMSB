# Module Exports

## MMSB/src/00_physical/mod.rs (00_physical)

Module `mod`

- `allocator_stats :: AllocatorStats`
- `gpu_memory_pool :: { GPUMemoryPool , PoolStats }`
- `nccl_integration :: { NCCLContext , NcclDataType , NcclRedOp }`

## MMSB/src/01_page/epoch.rs (01_page)

Module `epoch`

- `crate :: types :: { Epoch , EpochCell }`

## MMSB/src/01_page/mod.rs (01_page)

Module `mod`

- `allocator :: { PageAllocator , PageAllocatorConfig , PageInfo , PageSnapshotData }`
- `checkpoint :: { load_checkpoint , write_checkpoint }`
- `columnar_delta :: ColumnarDeltaBatch`
- `crate :: types :: { PageID , PageLocation , PageError , Epoch , EpochCell , DeltaID , Source , DeltaError }`
- `delta :: Delta`
- `delta_merge :: merge_deltas`
- `delta_validation :: validate_delta`
- `device :: DeviceRegistry`
- `device_registry :: DeviceBufferRegistry`
- `host_device_sync :: HostDeviceSync`
- `integrity_checker :: { DeltaIntegrityChecker , IntegrityReport , IntegrityViolation , IntegrityViolationKind }`
- `lockfree_allocator :: LockFreeAllocator`
- `page :: { Metadata , Page }`
- `replay_validator :: { ReplayCheckpoint , ReplayReport , ReplayValidator }`
- `simd_mask :: generate_mask`
- `tlog :: { LogSummary , TransactionLog , TransactionLogReader , summary }`

## MMSB/src/01_types/mod.rs (01_types)

Module `mod`

- `delta_types :: { DeltaID , Source , DeltaError }`
- `epoch_types :: { Epoch , EpochCell }`
- `gc :: { GCMetrics , MemoryPressureHandler }`
- `page_types :: { PageID , PageLocation , PageError }`

## MMSB/src/02_semiring/mod.rs (02_semiring)

Module `mod`

- `purity_validator :: { PurityFailure , PurityReport , PurityValidator }`
- `semiring_ops :: { accumulate , fold_add , fold_mul }`
- `semiring_types :: Semiring`
- `standard_semirings :: { BooleanSemiring , TropicalSemiring }`

## MMSB/src/03_dag/mod.rs (03_dag)

Module `mod`

- `cycle_detection :: has_cycle`
- `edge_types :: EdgeType`
- `graph_validator :: { GraphValidationReport , GraphValidator }`
- `shadow_graph :: { Edge , ShadowPageGraph }`
- `shadow_graph_traversal :: topological_sort`

## MMSB/src/03_dag/shadow_graph_mod.rs (03_dag)

Module `shadow_graph_mod`

- `super :: edge_types :: EdgeType`
- `super :: shadow_graph :: { Edge , ShadowPageGraph }`
- `super :: shadow_graph_traversal :: topological_sort`

## MMSB/src/04_propagation/mod.rs (04_propagation)

Module `mod`

- `propagation_command_buffer :: PropagationCommand`
- `propagation_engine :: PropagationEngine`
- `propagation_fastpath :: passthrough`
- `propagation_queue :: PropagationQueue`
- `ring_buffer :: LockFreeRingBuffer`
- `sparse_message_passing :: enqueue_sparse`
- `throughput_engine :: { ThroughputEngine , ThroughputMetrics }`
- `tick_orchestrator :: { TickMetrics , TickOrchestrator }`

## MMSB/src/05_adaptive/mod.rs (05_adaptive)

Module `mod`

- `locality_optimizer :: LocalityOptimizer`
- `memory_layout :: { MemoryLayout , AccessPattern , PageId , PhysAddr }`
- `page_clustering :: { PageCluster , PageClusterer }`

## MMSB/src/06_utility/mod.rs (06_utility)

Module `mod`

- `cpu_features :: CpuFeatures`
- `crate :: types :: GCMetrics`
- `invariant_checker :: { EpochMonotonicity , GraphAcyclicity , Invariant , InvariantChecker , InvariantContext , InvariantResult , PageConsistency }`
- `memory_monitor :: { MemoryMonitor , MemoryMonitorConfig , MemorySnapshot }`
- `provenance_tracker :: { ProvenanceResult , ProvenanceTracker }`
- `telemetry :: { Telemetry , TelemetrySnapshot }`

## MMSB/src/lib.rs (root)

Module `lib`

- `ffi :: *`

