# MMSB Code Structure Analysis


## 00_physical

### Rust

- fn `cudaMallocManaged` @ src/00_physical/allocator.rs:9
- fn `cudaFree` @ src/00_physical/allocator.rs:10
- struct `PageAllocatorConfig` @ src/00_physical/allocator.rs:14
- impl `Default` @ src/00_physical/allocator.rs:18
- fn `default` @ src/00_physical/allocator.rs:19
- struct `PageAllocator` @ src/00_physical/allocator.rs:27
- struct `PageInfo` @ src/00_physical/allocator.rs:34
- struct `PageSnapshotData` @ src/00_physical/allocator.rs:43
- impl `PageAllocator` @ src/00_physical/allocator.rs:52
- fn `new` @ src/00_physical/allocator.rs:53
- fn `allocate_raw` @ src/00_physical/allocator.rs:62
- fn `free` @ src/00_physical/allocator.rs:74
- fn `release` @ src/00_physical/allocator.rs:80
- fn `acquire_page` @ src/00_physical/allocator.rs:92
- fn `len` @ src/00_physical/allocator.rs:96
- fn `page_infos` @ src/00_physical/allocator.rs:100
- fn `snapshot_pages` @ src/00_physical/allocator.rs:114
- fn `restore_from_snapshot` @ src/00_physical/allocator.rs:129
- fn `test_page_info_metadata_roundtrip` @ src/00_physical/allocator.rs:186
- fn `test_unified_page` @ src/00_physical/allocator.rs:200
- fn `test_checkpoint_roundtrip_in_memory` @ src/00_physical/allocator.rs:223
- struct `AllocatorStats` @ src/00_physical/allocator_stats.rs:4
- impl `AllocatorStats` @ src/00_physical/allocator_stats.rs:9
- fn `record_alloc` @ src/00_physical/allocator_stats.rs:10
- fn `record_free` @ src/00_physical/allocator_stats.rs:14
- fn `snapshot` @ src/00_physical/allocator_stats.rs:18
- struct `DeviceRegistry` @ src/00_physical/device.rs:7
- impl `DeviceRegistry` @ src/00_physical/device.rs:11
- fn `register` @ src/00_physical/device.rs:12
- fn `unregister` @ src/00_physical/device.rs:16
- fn `get` @ src/00_physical/device.rs:20
- struct `DeviceBufferRegistry` @ src/00_physical/device_registry.rs:7
- impl `DeviceBufferRegistry` @ src/00_physical/device_registry.rs:11
- fn `insert` @ src/00_physical/device_registry.rs:12
- fn `remove` @ src/00_physical/device_registry.rs:16
- fn `len` @ src/00_physical/device_registry.rs:20
- fn `cudaMalloc` @ src/00_physical/gpu_memory_pool.rs:6
- fn `cudaFree` @ src/00_physical/gpu_memory_pool.rs:7
- struct `GPUMemoryPool` @ src/00_physical/gpu_memory_pool.rs:27
- struct `PoolStats` @ src/00_physical/gpu_memory_pool.rs:33
- impl `GPUMemoryPool` @ src/00_physical/gpu_memory_pool.rs:42
- fn `new` @ src/00_physical/gpu_memory_pool.rs:43
- fn `select_slab_size` @ src/00_physical/gpu_memory_pool.rs:59
- fn `allocate` @ src/00_physical/gpu_memory_pool.rs:66
- fn `deallocate` @ src/00_physical/gpu_memory_pool.rs:97
- fn `get_stats` @ src/00_physical/gpu_memory_pool.rs:111
- fn `clear` @ src/00_physical/gpu_memory_pool.rs:115
- impl `Drop` @ src/00_physical/gpu_memory_pool.rs:128
- fn `drop` @ src/00_physical/gpu_memory_pool.rs:129
- struct `HostDeviceSync` @ src/00_physical/host_device_sync.rs:4
- impl `HostDeviceSync` @ src/00_physical/host_device_sync.rs:8
- fn `enqueue` @ src/00_physical/host_device_sync.rs:9
- fn `drain` @ src/00_physical/host_device_sync.rs:13
- struct `LockFreeAllocator` @ src/00_physical/lockfree_allocator.rs:13
- impl `LockFreeAllocator` @ src/00_physical/lockfree_allocator.rs:20
- fn `new` @ src/00_physical/lockfree_allocator.rs:21
- fn `try_allocate_small` @ src/00_physical/lockfree_allocator.rs:30
- fn `deallocate_small` @ src/00_physical/lockfree_allocator.rs:66
- fn `get_stats` @ src/00_physical/lockfree_allocator.rs:100
- fn `clear` @ src/00_physical/lockfree_allocator.rs:108
- impl `Drop` @ src/00_physical/lockfree_allocator.rs:123
- fn `drop` @ src/00_physical/lockfree_allocator.rs:124
- fn `ncclGetUniqueId` @ src/00_physical/nccl_integration.rs:32
- fn `ncclCommInitRank` @ src/00_physical/nccl_integration.rs:33
- fn `ncclCommDestroy` @ src/00_physical/nccl_integration.rs:34
- fn `ncclAllReduce` @ src/00_physical/nccl_integration.rs:35
- fn `ncclAllGather` @ src/00_physical/nccl_integration.rs:44
- struct `NCCLCommunicator` @ src/00_physical/nccl_integration.rs:55
- struct `NCCLContext` @ src/00_physical/nccl_integration.rs:62
- impl `NCCLContext` @ src/00_physical/nccl_integration.rs:67
- fn `new` @ src/00_physical/nccl_integration.rs:68
- fn `init_communicator` @ src/00_physical/nccl_integration.rs:82
- fn `all_reduce` @ src/00_physical/nccl_integration.rs:102
- fn `all_gather` @ src/00_physical/nccl_integration.rs:122
- impl `Drop` @ src/00_physical/nccl_integration.rs:142
- fn `drop` @ src/00_physical/nccl_integration.rs:143

### Julia

- struct `CPUPropagationQueue` @ src/00_physical/DeviceFallback.jl:6
- struct `GPUCommandBuffer` @ src/00_physical/DeviceSync.jl:21
- fn `create_gpu_command_buffer` @ src/00_physical/DeviceSync.jl:38
- fn `enqueue_propagation_command` @ src/00_physical/DeviceSync.jl:60
- fn `wait_gpu_queue` @ src/00_physical/DeviceSync.jl:72
- fn `sync_page_to_gpu` @ src/00_physical/DeviceSync.jl:99
- fn `sync_page_to_cpu` @ src/00_physical/DeviceSync.jl:130
- fn `sync_bidirectional` @ src/00_physical/DeviceSync.jl:164
- fn `ensure_page_on_device` @ src/00_physical/DeviceSync.jl:194
- fn `async_sync_page_to_gpu` @ src/00_physical/DeviceSync.jl:222
- fn `batch_sync_to_gpu` @ src/00_physical/DeviceSync.jl:253
- fn `batch_sync_to_cpu` @ src/00_physical/DeviceSync.jl:280
- fn `get_sync_statistics` @ src/00_physical/DeviceSync.jl:309
- fn `prefetch_pages_to_gpu` @ src/00_physical/DeviceSync.jl:351
- fn `delta_merge_kernel` @ src/00_physical/GPUKernels.jl:33
- fn `launch_delta_merge` @ src/00_physical/GPUKernels.jl:62
- fn `page_copy_kernel` @ src/00_physical/GPUKernels.jl:83
- fn `launch_page_copy` @ src/00_physical/GPUKernels.jl:100
- fn `page_zero_kernel` @ src/00_physical/GPUKernels.jl:113
- fn `launch_page_zero` @ src/00_physical/GPUKernels.jl:126
- fn `page_compare_kernel` @ src/00_physical/GPUKernels.jl:143
- fn `launch_page_compare` @ src/00_physical/GPUKernels.jl:160
- fn `sparse_delta_apply_kernel` @ src/00_physical/GPUKernels.jl:187
- fn `launch_sparse_delta_apply` @ src/00_physical/GPUKernels.jl:206
- fn `compute_optimal_kernel_config` @ src/00_physical/GPUKernels.jl:230
- fn `create_page` @ src/00_physical/PageAllocator.jl:22
- fn `delete_page` @ src/00_physical/PageAllocator.jl:49
- fn `migrate_page` @ src/00_physical/PageAllocator.jl:72
- fn `resize_page` @ src/00_physical/PageAllocator.jl:86
- fn `allocate_page_arrays` @ src/00_physical/PageAllocator.jl:99
- fn `clone_page` @ src/00_physical/PageAllocator.jl:106
- struct `GPUMemoryPool` @ src/00_physical/UnifiedMemory.jl:22
- fn `GPUMemoryPool` @ src/00_physical/UnifiedMemory.jl:39
- fn `allocate_from_pool` @ src/00_physical/UnifiedMemory.jl:54
- fn `deallocate_to_pool` @ src/00_physical/UnifiedMemory.jl:66
- fn `get_pool_stats` @ src/00_physical/UnifiedMemory.jl:76
- fn `is_unified_memory_available` @ src/00_physical/UnifiedMemory.jl:99
- fn `create_unified_page` @ src/00_physical/UnifiedMemory.jl:128
- fn `prefetch_unified_to_gpu` @ src/00_physical/UnifiedMemory.jl:146
- fn `prefetch_unified_to_cpu` @ src/00_physical/UnifiedMemory.jl:168
- fn `adaptive_prefetch_distance` @ src/00_physical/UnifiedMemory.jl:194
- fn `set_preferred_location` @ src/00_physical/UnifiedMemory.jl:225
- fn `convert_to_unified` @ src/00_physical/UnifiedMemory.jl:249
- fn `enable_read_mostly_hint` @ src/00_physical/UnifiedMemory.jl:278
- fn `disable_read_mostly_hint` @ src/00_physical/UnifiedMemory.jl:292

## 01_page

### Rust

- fn `write_checkpoint` @ src/01_page/checkpoint.rs:11
- fn `load_checkpoint` @ src/01_page/checkpoint.rs:41
- struct `DeltaID` @ src/01_page/delta.rs:7
- struct `Source` @ src/01_page/delta.rs:10
- struct `Delta` @ src/01_page/delta.rs:13
- impl `Delta` @ src/01_page/delta.rs:25
- fn `new_dense` @ src/01_page/delta.rs:26
- fn `new_sparse` @ src/01_page/delta.rs:54
- fn `merge` @ src/01_page/delta.rs:83
- fn `to_dense` @ src/01_page/delta.rs:122
- fn `apply_to` @ src/01_page/delta.rs:138
- fn `now_ns` @ src/01_page/delta.rs:171
- fn `merge_deltas` @ src/01_page/delta_merge.rs:7
- fn `merge_dense_simd` @ src/01_page/delta_merge.rs:124
- fn `validate_delta` @ src/01_page/delta_validation.rs:4
- struct `Epoch` @ src/01_page/epoch.rs:6
- impl `Epoch` @ src/01_page/epoch.rs:8
- fn `new` @ src/01_page/epoch.rs:9
- struct `EpochCell` @ src/01_page/epoch.rs:16
- impl `EpochCell` @ src/01_page/epoch.rs:20
- fn `new` @ src/01_page/epoch.rs:21
- fn `load` @ src/01_page/epoch.rs:28
- fn `store` @ src/01_page/epoch.rs:33
- fn `increment` @ src/01_page/epoch.rs:38
- fn `cudaMallocManaged` @ src/01_page/page.rs:18
- fn `cudaFree` @ src/01_page/page.rs:19
- impl `PageLocation` @ src/01_page/page.rs:34
- fn `from_tag` @ src/01_page/page.rs:35
- struct `PageID` @ src/01_page/page.rs:48
- impl `fmt` @ src/01_page/page.rs:50
- fn `fmt` @ src/01_page/page.rs:51
- struct `Metadata` @ src/01_page/page.rs:58
- impl `Metadata` @ src/01_page/page.rs:62
- fn `new` @ src/01_page/page.rs:63
- fn `insert` @ src/01_page/page.rs:69
- fn `clone_store` @ src/01_page/page.rs:76
- fn `from_entries` @ src/01_page/page.rs:80
- struct `Page` @ src/01_page/page.rs:89
- impl `Page` @ src/01_page/page.rs:101
- fn `new` @ src/01_page/page.rs:102
- fn `size` @ src/01_page/page.rs:153
- fn `location` @ src/01_page/page.rs:157
- fn `data_slice` @ src/01_page/page.rs:161
- fn `data_mut_slice` @ src/01_page/page.rs:165
- fn `mask_slice` @ src/01_page/page.rs:169
- fn `data_ptr` @ src/01_page/page.rs:173
- fn `mask_ptr` @ src/01_page/page.rs:177
- fn `epoch` @ src/01_page/page.rs:181
- fn `set_epoch` @ src/01_page/page.rs:185
- fn `metadata_entries` @ src/01_page/page.rs:189
- fn `set_metadata` @ src/01_page/page.rs:193
- fn `metadata_blob` @ src/01_page/page.rs:197
- fn `set_metadata_blob` @ src/01_page/page.rs:214
- fn `apply_delta` @ src/01_page/page.rs:235
- impl `Clone` @ src/01_page/page.rs:284
- fn `clone` @ src/01_page/page.rs:285
- impl `Drop` @ src/01_page/page.rs:313
- fn `drop` @ src/01_page/page.rs:314
- fn `read_u32` @ src/01_page/page.rs:356
- fn `read_bytes` @ src/01_page/page.rs:367
- fn `allocate_zeroed` @ src/01_page/page.rs:376
- fn `generate_mask` @ src/01_page/simd_mask.rs:2
- struct `TransactionLog` @ src/01_page/tlog.rs:12
- struct `TransactionLogReader` @ src/01_page/tlog.rs:19
- struct `LogSummary` @ src/01_page/tlog.rs:25
- impl `TransactionLog` @ src/01_page/tlog.rs:31
- fn `new` @ src/01_page/tlog.rs:32
- fn `append` @ src/01_page/tlog.rs:48
- fn `len` @ src/01_page/tlog.rs:64
- fn `drain` @ src/01_page/tlog.rs:68
- fn `current_offset` @ src/01_page/tlog.rs:73
- impl `TransactionLogReader` @ src/01_page/tlog.rs:88
- fn `open` @ src/01_page/tlog.rs:89
- fn `next` @ src/01_page/tlog.rs:96
- fn `free` @ src/01_page/tlog.rs:100
- impl `Drop` @ src/01_page/tlog.rs:103
- fn `drop` @ src/01_page/tlog.rs:104
- fn `summary` @ src/01_page/tlog.rs:107
- fn `serialize_frame` @ src/01_page/tlog.rs:135
- fn `read_frame` @ src/01_page/tlog.rs:167
- fn `validate_header` @ src/01_page/tlog.rs:235
- fn `encode_rle` @ src/01_page/tlog_compression.rs:4
- fn `decode_rle` @ src/01_page/tlog_compression.rs:26
- fn `bitpack_mask` @ src/01_page/tlog_compression.rs:43
- fn `bitunpack_mask` @ src/01_page/tlog_compression.rs:55
- struct `CompressionStats` @ src/01_page/tlog_compression.rs:72
- fn `compress_delta_mask` @ src/01_page/tlog_compression.rs:78
- fn `compact` @ src/01_page/tlog_compression.rs:96
- fn `apply_log` @ src/01_page/tlog_replay.rs:3
- fn `read_log` @ src/01_page/tlog_serialization.rs:6

### Julia

- struct `Delta` @ src/01_page/Delta.jl:11
- fn `Delta` @ src/01_page/Delta.jl:24
- fn `Delta` @ src/01_page/Delta.jl:36
- fn `new_delta_handle` @ src/01_page/Delta.jl:55
- fn `apply_delta` @ src/01_page/Delta.jl:61
- fn `dense_data` @ src/01_page/Delta.jl:65
- fn `serialize_delta` @ src/01_page/Delta.jl:82
- fn `deserialize_delta` @ src/01_page/Delta.jl:100
- fn `set_intent_metadata` @ src/01_page/Delta.jl:110
- fn `intent_metadata` @ src/01_page/Delta.jl:130
- fn `_encode_metadata_value` @ src/01_page/Delta.jl:136
- fn `_encode_metadata_dict` @ src/01_page/Delta.jl:160
- fn `_escape_metadata_string` @ src/01_page/Delta.jl:169
- fn `merge_deltas_simd` @ src/01_page/Delta.jl:194
- struct `_MetadataParser` @ src/01_page/Delta.jl:219
- fn `_decode_metadata` @ src/01_page/Delta.jl:224
- fn `_parse_metadata_value` @ src/01_page/Delta.jl:231
- fn `_parse_metadata_object` @ src/01_page/Delta.jl:254
- fn `_parse_metadata_array` @ src/01_page/Delta.jl:279
- fn `_parse_metadata_string` @ src/01_page/Delta.jl:300
- fn `_parse_metadata_number` @ src/01_page/Delta.jl:329
- fn `_skip_ws` @ src/01_page/Delta.jl:348
- fn `_consume` @ src/01_page/Delta.jl:354
- fn `_peek` @ src/01_page/Delta.jl:360
- struct `Page` @ src/01_page/Page.jl:17
- fn `Page` @ src/01_page/Page.jl:27
- fn `initialize` @ src/01_page/Page.jl:46
- fn `activate` @ src/01_page/Page.jl:52
- fn `deactivate` @ src/01_page/Page.jl:58
- fn `read_page` @ src/01_page/Page.jl:64
- fn `_apply_metadata` @ src/01_page/Page.jl:73
- fn `_encode_metadata_dict` @ src/01_page/Page.jl:81
- fn `_coerce_metadata_value` @ src/01_page/Page.jl:96
- fn `_decode_metadata_blob` @ src/01_page/Page.jl:102
- fn `_blank_state_like` @ src/01_page/ReplayEngine.jl:13
- fn `_apply_delta` @ src/01_page/ReplayEngine.jl:48
- fn `_all_deltas` @ src/01_page/ReplayEngine.jl:54
- fn `replay_to_epoch` @ src/01_page/ReplayEngine.jl:58
- fn `replay_to_timestamp` @ src/01_page/ReplayEngine.jl:71
- fn `replay_from_checkpoint` @ src/01_page/ReplayEngine.jl:84
- fn `replay_page_history` @ src/01_page/ReplayEngine.jl:92
- fn `verify_state_consistency` @ src/01_page/ReplayEngine.jl:115
- fn `replay_with_predicate` @ src/01_page/ReplayEngine.jl:129
- fn `incremental_replay` @ src/01_page/ReplayEngine.jl:137
- fn `compress_delta_mask` @ src/01_page/TLog.jl:19
- fn `_with_rust_errors` @ src/01_page/TLog.jl:28
- fn `append_to_log` @ src/01_page/TLog.jl:39
- fn `log_summary` @ src/01_page/TLog.jl:48
- fn `_iterate_log` @ src/01_page/TLog.jl:56
- fn `query_log` @ src/01_page/TLog.jl:66
- fn `get_deltas_in_range` @ src/01_page/TLog.jl:97
- fn `compute_log_statistics` @ src/01_page/TLog.jl:105
- fn `replay_log` @ src/01_page/TLog.jl:114
- fn `checkpoint_log` @ src/01_page/TLog.jl:120
- fn `load_checkpoint` @ src/01_page/TLog.jl:128
- fn `_refresh_pages` @ src/01_page/TLog.jl:137

## 01_types


### Julia

- struct `PageNotFoundError` @ src/01_types/Errors.jl:15
- struct `InvalidDeltaError` @ src/01_types/Errors.jl:20
- struct `GPUMemoryError` @ src/01_types/Errors.jl:25
- struct `SerializationError` @ src/01_types/Errors.jl:29
- struct `GraphCycleError` @ src/01_types/Errors.jl:33
- struct `UnsupportedLocationError` @ src/01_types/Errors.jl:38
- struct `MMSBConfig` @ src/01_types/MMSBState.jl:25
- fn `MMSBConfig` @ src/01_types/MMSBState.jl:34
- struct `MMSBState` @ src/01_types/MMSBState.jl:51
- fn `MMSBState` @ src/01_types/MMSBState.jl:61
- fn `allocate_page_id` @ src/01_types/MMSBState.jl:91
- fn `allocate_delta_id` @ src/01_types/MMSBState.jl:111
- fn `get_page` @ src/01_types/MMSBState.jl:124
- fn `register_page` @ src/01_types/MMSBState.jl:135

## 02_semiring

### Rust

- fn `fold_add` @ src/02_semiring/semiring_ops.rs:4
- fn `fold_mul` @ src/02_semiring/semiring_ops.rs:11
- fn `accumulate` @ src/02_semiring/semiring_ops.rs:18
- fn `zero` @ src/02_semiring/semiring_types.rs:6
- fn `one` @ src/02_semiring/semiring_types.rs:9
- fn `add` @ src/02_semiring/semiring_types.rs:12
- fn `mul` @ src/02_semiring/semiring_types.rs:15
- struct `TropicalSemiring` @ src/02_semiring/standard_semirings.rs:4
- impl `Semiring` @ src/02_semiring/standard_semirings.rs:6
- fn `zero` @ src/02_semiring/standard_semirings.rs:9
- fn `one` @ src/02_semiring/standard_semirings.rs:13
- fn `add` @ src/02_semiring/standard_semirings.rs:17
- fn `mul` @ src/02_semiring/standard_semirings.rs:21
- struct `BooleanSemiring` @ src/02_semiring/standard_semirings.rs:27
- impl `Semiring` @ src/02_semiring/standard_semirings.rs:29
- fn `zero` @ src/02_semiring/standard_semirings.rs:32
- fn `one` @ src/02_semiring/standard_semirings.rs:36
- fn `add` @ src/02_semiring/standard_semirings.rs:40
- fn `mul` @ src/02_semiring/standard_semirings.rs:44

### Julia

- fn `route_delta` @ src/02_semiring/DeltaRouter.jl:29
- fn `create_delta` @ src/02_semiring/DeltaRouter.jl:47
- fn `batch_route_deltas` @ src/02_semiring/DeltaRouter.jl:71
- fn `propagate_change` @ src/02_semiring/DeltaRouter.jl:89
- fn `propagate_change` @ src/02_semiring/DeltaRouter.jl:94
- struct `SemiringOps` @ src/02_semiring/Semiring.jl:9
- fn `tropical_semiring` @ src/02_semiring/Semiring.jl:16
- fn `boolean_semiring` @ src/02_semiring/Semiring.jl:25
- fn `_bool_buf` @ src/02_semiring/Semiring.jl:36
- fn `tropical_fold_add` @ src/02_semiring/Semiring.jl:49
- fn `tropical_fold_mul` @ src/02_semiring/Semiring.jl:57
- fn `tropical_accumulate` @ src/02_semiring/Semiring.jl:65
- fn `boolean_fold_add` @ src/02_semiring/Semiring.jl:72
- fn `boolean_fold_mul` @ src/02_semiring/Semiring.jl:80
- fn `boolean_accumulate` @ src/02_semiring/Semiring.jl:88
- struct `SemiringConfigOptions` @ src/02_semiring/SemiringConfig.jl:5
- fn `build_semiring` @ src/02_semiring/SemiringConfig.jl:9

## 03_dag

### Rust

- fn `has_cycle` @ src/03_dag/cycle_detection.rs:11
- fn `dfs` @ src/03_dag/cycle_detection.rs:15
- struct `Edge` @ src/03_dag/shadow_graph.rs:7
- struct `ShadowPageGraph` @ src/03_dag/shadow_graph.rs:14
- impl `ShadowPageGraph` @ src/03_dag/shadow_graph.rs:18
- fn `add_edge` @ src/03_dag/shadow_graph.rs:19
- fn `remove_edge` @ src/03_dag/shadow_graph.rs:24
- fn `descendants` @ src/03_dag/shadow_graph.rs:30
- fn `topological_sort` @ src/03_dag/shadow_graph_traversal.rs:5

### Julia

- fn `add_edge` @ src/03_dag/DependencyGraph.jl:33
- fn `remove_edge` @ src/03_dag/DependencyGraph.jl:69
- fn `has_edge` @ src/03_dag/DependencyGraph.jl:88
- fn `get_children` @ src/03_dag/DependencyGraph.jl:106
- fn `get_parents` @ src/03_dag/DependencyGraph.jl:120
- fn `find_descendants` @ src/03_dag/DependencyGraph.jl:137
- fn `find_ancestors` @ src/03_dag/DependencyGraph.jl:167
- fn `detect_cycles` @ src/03_dag/DependencyGraph.jl:201
- fn `dfs_cycle_detect` @ src/03_dag/DependencyGraph.jl:236
- fn `topological_order` @ src/03_dag/DependencyGraph.jl:280
- fn `reverse_postorder` @ src/03_dag/DependencyGraph.jl:339
- fn `dfs_postorder` @ src/03_dag/DependencyGraph.jl:344
- fn `compute_closure` @ src/03_dag/DependencyGraph.jl:373
- struct `EventSubscription` @ src/03_dag/EventSystem.jl:52
- fn `EventSubscription` @ src/03_dag/EventSystem.jl:57
- fn `emit_event` @ src/03_dag/EventSystem.jl:78
- fn `subscribe` @ src/03_dag/EventSystem.jl:102
- fn `unsubscribe` @ src/03_dag/EventSystem.jl:118
- fn `log_event` @ src/03_dag/EventSystem.jl:129
- fn `clear_subscriptions` @ src/03_dag/EventSystem.jl:138
- fn `get_subscription_count` @ src/03_dag/EventSystem.jl:150
- fn `create_debug_subscriber` @ src/03_dag/EventSystem.jl:161
- fn `create_logging_subscriber` @ src/03_dag/EventSystem.jl:182
- fn `_serialize_event` @ src/03_dag/EventSystem.jl:190
- fn `log_event_to_page` @ src/03_dag/EventSystem.jl:202
- struct `ShadowPageGraph` @ src/03_dag/ShadowPageGraph.jl:31
- fn `ShadowPageGraph` @ src/03_dag/ShadowPageGraph.jl:36
- fn `_ensure_vertex` @ src/03_dag/ShadowPageGraph.jl:48
- fn `add_dependency` @ src/03_dag/ShadowPageGraph.jl:63
- fn `remove_dependency` @ src/03_dag/ShadowPageGraph.jl:86
- fn `get_children` @ src/03_dag/ShadowPageGraph.jl:102
- fn `get_parents` @ src/03_dag/ShadowPageGraph.jl:113
- fn `_dfs_has_cycle` @ src/03_dag/ShadowPageGraph.jl:122
- fn `has_cycle` @ src/03_dag/ShadowPageGraph.jl:141
- fn `_all_vertices` @ src/03_dag/ShadowPageGraph.jl:154
- fn `topological_sort` @ src/03_dag/ShadowPageGraph.jl:170

## 04_propagation

### Rust

- struct `PropagationCommand` @ src/04_propagation/propagation_command_buffer.rs:5
- struct `PropagationEngine` @ src/04_propagation/propagation_engine.rs:10
- impl `Default` @ src/04_propagation/propagation_engine.rs:15
- fn `default` @ src/04_propagation/propagation_engine.rs:16
- impl `PropagationEngine` @ src/04_propagation/propagation_engine.rs:24
- fn `register_callback` @ src/04_propagation/propagation_engine.rs:25
- fn `enqueue` @ src/04_propagation/propagation_engine.rs:29
- fn `drain` @ src/04_propagation/propagation_engine.rs:33
- fn `passthrough` @ src/04_propagation/propagation_fastpath.rs:4
- struct `PropagationQueue` @ src/04_propagation/propagation_queue.rs:8
- impl `PropagationQueue` @ src/04_propagation/propagation_queue.rs:13
- fn `new` @ src/04_propagation/propagation_queue.rs:14
- fn `push` @ src/04_propagation/propagation_queue.rs:21
- fn `pop` @ src/04_propagation/propagation_queue.rs:26
- fn `push_batch` @ src/04_propagation/propagation_queue.rs:34
- fn `drain_batch` @ src/04_propagation/propagation_queue.rs:40
- fn `is_empty` @ src/04_propagation/propagation_queue.rs:48
- fn `len` @ src/04_propagation/propagation_queue.rs:52
- fn `enqueue_sparse` @ src/04_propagation/sparse_message_passing.rs:5

### Julia

- struct `CUDAGraphState` @ src/04_propagation/PropagationEngine.jl:35
- struct `PropagationQueue` @ src/04_propagation/PropagationEngine.jl:44
- fn `enable_graph_capture` @ src/04_propagation/PropagationEngine.jl:57
- fn `disable_graph_capture` @ src/04_propagation/PropagationEngine.jl:69
- fn `replay_cuda_graph` @ src/04_propagation/PropagationEngine.jl:90
- fn `batch_route_deltas` @ src/04_propagation/PropagationEngine.jl:111
- fn `_buffer` @ src/04_propagation/PropagationEngine.jl:146
- fn `register_recompute_fn` @ src/04_propagation/PropagationEngine.jl:158
- fn `register_passthrough_recompute` @ src/04_propagation/PropagationEngine.jl:171
- fn `queue_recomputation` @ src/04_propagation/PropagationEngine.jl:185
- fn `propagate_change` @ src/04_propagation/PropagationEngine.jl:198
- fn `propagate_change` @ src/04_propagation/PropagationEngine.jl:203
- fn `_aggregate_children` @ src/04_propagation/PropagationEngine.jl:213
- fn `_execute_command_buffer` @ src/04_propagation/PropagationEngine.jl:226
- fn `_apply_edges` @ src/04_propagation/PropagationEngine.jl:234
- fn `_handle_data_dependency` @ src/04_propagation/PropagationEngine.jl:247
- fn `_collect_descendants` @ src/04_propagation/PropagationEngine.jl:260
- fn `schedule_propagation` @ src/04_propagation/PropagationEngine.jl:280
- fn `execute_propagation` @ src/04_propagation/PropagationEngine.jl:296
- fn `recompute_page` @ src/04_propagation/PropagationEngine.jl:310
- fn `mark_page_stale` @ src/04_propagation/PropagationEngine.jl:334
- fn `schedule_gpu_sync` @ src/04_propagation/PropagationEngine.jl:345
- fn `invalidate_compilation` @ src/04_propagation/PropagationEngine.jl:355
- fn `topological_order_subset` @ src/04_propagation/PropagationEngine.jl:368
- fn `schedule` @ src/04_propagation/PropagationScheduler.jl:8
- struct `Transaction` @ src/04_propagation/TransactionIsolation.jl:5

## 05_adaptive

### Rust

- struct `PageEdge` @ src/05_adaptive/locality_optimizer.rs:10
- struct `LocalityOptimizer` @ src/05_adaptive/locality_optimizer.rs:17
- impl `LocalityOptimizer` @ src/05_adaptive/locality_optimizer.rs:24
- fn `new` @ src/05_adaptive/locality_optimizer.rs:25
- fn `add_edge` @ src/05_adaptive/locality_optimizer.rs:32
- fn `compute_ordering` @ src/05_adaptive/locality_optimizer.rs:38
- fn `dfs_visit` @ src/05_adaptive/locality_optimizer.rs:78
- fn `assign_addresses` @ src/05_adaptive/locality_optimizer.rs:102
- fn `test_locality_optimizer` @ src/05_adaptive/locality_optimizer.rs:116
- struct `MemoryLayout` @ src/05_adaptive/memory_layout.rs:16
- struct `AccessPattern` @ src/05_adaptive/memory_layout.rs:25
- impl `MemoryLayout` @ src/05_adaptive/memory_layout.rs:30
- fn `new` @ src/05_adaptive/memory_layout.rs:32
- fn `locality_cost` @ src/05_adaptive/memory_layout.rs:40
- fn `optimize_layout` @ src/05_adaptive/memory_layout.rs:56
- fn `test_memory_layout_creation` @ src/05_adaptive/memory_layout.rs:86
- fn `test_locality_cost_empty` @ src/05_adaptive/memory_layout.rs:93
- fn `test_optimize_layout` @ src/05_adaptive/memory_layout.rs:102
- struct `PageCluster` @ src/05_adaptive/page_clustering.rs:9
- struct `PageClusterer` @ src/05_adaptive/page_clustering.rs:18
- impl `PageClusterer` @ src/05_adaptive/page_clustering.rs:25
- fn `new` @ src/05_adaptive/page_clustering.rs:26
- fn `cluster_pages` @ src/05_adaptive/page_clustering.rs:34
- fn `clusters` @ src/05_adaptive/page_clustering.rs:86
- fn `test_page_clustering` @ src/05_adaptive/page_clustering.rs:96

### Julia

- struct `LayoutState` @ src/05_adaptive/AdaptiveLayout.jl:22
- fn `LayoutState` @ src/05_adaptive/AdaptiveLayout.jl:29
- fn `optimize_layout` @ src/05_adaptive/AdaptiveLayout.jl:39
- fn `compute_locality_score` @ src/05_adaptive/AdaptiveLayout.jl:72
- fn `compute_entropy` @ src/05_adaptive/EntropyReduction.jl:19
- fn `reduce_entropy` @ src/05_adaptive/EntropyReduction.jl:42
- fn `entropy_gradient` @ src/05_adaptive/EntropyReduction.jl:63
- struct `EdgeRewrite` @ src/05_adaptive/GraphRewriting.jl:21
- fn `rewrite_dag` @ src/05_adaptive/GraphRewriting.jl:33
- fn `can_reorder` @ src/05_adaptive/GraphRewriting.jl:71
- fn `compute_edge_cost` @ src/05_adaptive/GraphRewriting.jl:84
- struct `AccessTrace` @ src/05_adaptive/LocalityAnalysis.jl:16
- struct `LocalityMetrics` @ src/05_adaptive/LocalityAnalysis.jl:26
- fn `analyze_locality` @ src/05_adaptive/LocalityAnalysis.jl:38
- fn `compute_reuse_distance` @ src/05_adaptive/LocalityAnalysis.jl:77

## 06_utility

### Rust

- struct `CpuFeatures` @ src/06_utility/cpu_features.rs:4
- impl `CpuFeatures` @ src/06_utility/cpu_features.rs:13
- fn `detect` @ src/06_utility/cpu_features.rs:14
- fn `get` @ src/06_utility/cpu_features.rs:36
- struct `Telemetry` @ src/06_utility/telemetry.rs:10
- struct `TelemetrySnapshot` @ src/06_utility/telemetry.rs:29
- impl `Telemetry` @ src/06_utility/telemetry.rs:39
- fn `new` @ src/06_utility/telemetry.rs:41
- fn `record_cache_miss` @ src/06_utility/telemetry.rs:54
- fn `record_cache_hit` @ src/06_utility/telemetry.rs:59
- fn `record_allocation` @ src/06_utility/telemetry.rs:64
- fn `record_propagation` @ src/06_utility/telemetry.rs:70
- fn `snapshot` @ src/06_utility/telemetry.rs:76
- fn `reset` @ src/06_utility/telemetry.rs:89
- impl `Default` @ src/06_utility/telemetry.rs:99
- fn `default` @ src/06_utility/telemetry.rs:100
- impl `TelemetrySnapshot` @ src/06_utility/telemetry.rs:105
- fn `cache_hit_rate` @ src/06_utility/telemetry.rs:107
- fn `avg_propagation_latency_us` @ src/06_utility/telemetry.rs:117
- fn `avg_allocation_size` @ src/06_utility/telemetry.rs:126
- fn `test_telemetry_basic` @ src/06_utility/telemetry.rs:140
- fn `test_cache_hit_rate` @ src/06_utility/telemetry.rs:154
- fn `test_reset` @ src/06_utility/telemetry.rs:166

### Julia

- struct `WeightedCost` @ src/06_utility/CostAggregation.jl:16
- fn `aggregate_costs` @ src/06_utility/CostAggregation.jl:27
- fn `normalize_costs` @ src/06_utility/CostAggregation.jl:37
- struct `RetryPolicy` @ src/06_utility/ErrorRecovery.jl:5
- fn `retry_with_backoff` @ src/06_utility/ErrorRecovery.jl:15
- struct `LRUTracker` @ src/06_utility/MemoryPressure.jl:5
- fn `evict_lru_pages` @ src/06_utility/MemoryPressure.jl:13
- struct `MMSBStats` @ src/06_utility/Monitoring.jl:17
- fn `track_delta_latency` @ src/06_utility/Monitoring.jl:40
- fn `track_propagation_latency` @ src/06_utility/Monitoring.jl:45
- fn `compute_graph_depth` @ src/06_utility/Monitoring.jl:50
- fn `_dfs_depth` @ src/06_utility/Monitoring.jl:59
- fn `get_stats` @ src/06_utility/Monitoring.jl:74
- fn `print_stats` @ src/06_utility/Monitoring.jl:108
- fn `reset_stats` @ src/06_utility/Monitoring.jl:119
- struct `CostComponents` @ src/06_utility/cost_functions.jl:18
- fn `compute_cache_cost` @ src/06_utility/cost_functions.jl:30
- fn `compute_memory_cost` @ src/06_utility/cost_functions.jl:45
- fn `compute_latency_cost` @ src/06_utility/cost_functions.jl:59
- fn `from_telemetry` @ src/06_utility/cost_functions.jl:73
- struct `PageDistribution` @ src/06_utility/entropy_measure.jl:18
- fn `PageDistribution` @ src/06_utility/entropy_measure.jl:23
- fn `compute_entropy` @ src/06_utility/entropy_measure.jl:32
- fn `state_entropy` @ src/06_utility/entropy_measure.jl:51
- fn `entropy_reduction` @ src/06_utility/entropy_measure.jl:71
- struct `UtilityState` @ src/06_utility/utility_engine.jl:20
- fn `UtilityState` @ src/06_utility/utility_engine.jl:27
- fn `compute_utility` @ src/06_utility/utility_engine.jl:44
- fn `update_utility` @ src/06_utility/utility_engine.jl:58
- fn `utility_trend` @ src/06_utility/utility_engine.jl:75

## root

### Rust

- fn `main` @ build.rs:1
- struct `PageHandle` @ src/ffi.rs:19
- impl `PageHandle` @ src/ffi.rs:23
- fn `null` @ src/ffi.rs:24
- struct `DeltaHandle` @ src/ffi.rs:33
- impl `DeltaHandle` @ src/ffi.rs:37
- fn `null` @ src/ffi.rs:38
- struct `AllocatorHandle` @ src/ffi.rs:47
- impl `AllocatorHandle` @ src/ffi.rs:51
- fn `null` @ src/ffi.rs:52
- struct `TLogHandle` @ src/ffi.rs:61
- impl `TLogHandle` @ src/ffi.rs:65
- fn `null` @ src/ffi.rs:66
- struct `TLogReaderHandle` @ src/ffi.rs:75
- impl `TLogReaderHandle` @ src/ffi.rs:79
- fn `null` @ src/ffi.rs:80
- impl `MMSBErrorCode` @ src/ffi.rs:104
- fn `is_retryable` @ src/ffi.rs:105
- fn `is_fatal` @ src/ffi.rs:113
- fn `set_last_error` @ src/ffi.rs:127
- fn `log_error_code` @ src/ffi.rs:133
- struct `TLogSummary` @ src/ffi.rs:163
- struct `PageInfoABI` @ src/ffi.rs:171
- struct `EpochABI` @ src/ffi.rs:182
- impl `From` @ src/ffi.rs:186
- fn `from` @ src/ffi.rs:187
- fn `convert_location` @ src/ffi.rs:192
- struct `SemiringPairF64` @ src/ffi.rs:198
- struct `SemiringPairBool` @ src/ffi.rs:205
- fn `mask_from_bytes` @ src/ffi.rs:210
- fn `vec_from_ptr` @ src/ffi.rs:237
- fn `slice_from_ptr` @ src/ffi.rs:244
- fn `dense_delta` @ tests/delta_validation.rs:3
- fn `validates_dense_lengths` @ tests/delta_validation.rs:18
- fn `rejects_mismatched_dense_lengths` @ tests/delta_validation.rs:24
- fn `example_page_allocation` @ tests/examples_basic.rs:5
- fn `example_delta_operations` @ tests/examples_basic.rs:17
- fn `example_checkpoint` @ tests/examples_basic.rs:27
- fn `read_page` @ tests/mmsb_tests.rs:9
- fn `test_page_info_metadata_roundtrip` @ tests/mmsb_tests.rs:14
- fn `test_page_snapshot_and_restore` @ tests/mmsb_tests.rs:26
- fn `test_thread_safe_allocator` @ tests/mmsb_tests.rs:43
- fn `test_gpu_delta_kernels` @ tests/mmsb_tests.rs:66
- fn `test_checkpoint_log_and_restore` @ tests/mmsb_tests.rs:69
- fn `test_invalid_page_deletion_is_safe` @ tests/mmsb_tests.rs:106
- fn `test_sparse_delta_application` @ tests/mmsb_tests.rs:113
- fn `test_dense_delta_application` @ tests/mmsb_tests.rs:139
- fn `test_api_public_interface` @ tests/mmsb_tests.rs:163
- fn `test_allocator_cpu_gpu_latency` @ tests/week27_31_integration.rs:7
- fn `test_semiring_operations_tropical` @ tests/week27_31_integration.rs:19
- fn `test_delta_merge_simd` @ tests/week27_31_integration.rs:25
- fn `test_lockfree_allocator` @ tests/week27_31_integration.rs:45
- fn `test_propagation_queue` @ tests/week27_31_integration.rs:58
- fn `test_cpu_features` @ tests/week27_31_integration.rs:65

### Julia

- fn `_start_state` @ benchmark/benchmarks.jl:43
- fn `_stop_state` @ benchmark/benchmarks.jl:47
- fn `_page` @ benchmark/benchmarks.jl:53
- fn `_populate_pages` @ benchmark/benchmarks.jl:57
- fn `_seed_pages` @ benchmark/benchmarks.jl:61
- fn `_replay_sequence` @ benchmark/benchmarks.jl:68
- fn `_stress_updates` @ benchmark/benchmarks.jl:74
- fn `_link_chain` @ benchmark/benchmarks.jl:81
- fn `_checkpoint` @ benchmark/benchmarks.jl:89
- fn `_measure_ns` @ benchmark/benchmarks.jl:95
- fn `_graph_fixture` @ benchmark/benchmarks.jl:101
- fn `_graph_bfs` @ benchmark/benchmarks.jl:117
- fn `_build_batch_deltas` @ benchmark/benchmarks.jl:131
- fn `_full_system_benchmark` @ benchmark/benchmarks.jl:141
- fn `_trial_to_dict` @ benchmark/benchmarks.jl:412
- fn `_select_suite` @ benchmark/benchmarks.jl:423
- fn `_collect_instrumentation_report` @ benchmark/benchmarks.jl:435
- fn `_to_mutable` @ benchmark/benchmarks.jl:472
- fn `run_benchmarks` @ benchmark/benchmarks.jl:482
- fn `compare_with_baseline` @ benchmark/benchmarks.jl:518
- fn `_format_time` @ benchmark/helpers.jl:6
- fn `_format_bytes` @ benchmark/helpers.jl:18
- fn `analyze_results` @ benchmark/helpers.jl:30
- fn `check_performance_targets` @ benchmark/helpers.jl:51
- fn `mmsb_start` @ src/API.jl:30
- fn `mmsb_stop` @ src/API.jl:44
- fn `_resolve_location` @ src/API.jl:52
- fn `create_page` @ src/API.jl:68
- fn `update_page` @ src/API.jl:88
- fn `query_page` @ src/API.jl:113
- fn `register_error_hook` @ src/ffi/FFIWrapper.jl:9
- fn `_check_rust_error` @ src/ffi/FFIWrapper.jl:14
- struct `RustPageHandle` @ src/ffi/FFIWrapper.jl:19
- struct `RustDeltaHandle` @ src/ffi/FFIWrapper.jl:23
- struct `RustAllocatorHandle` @ src/ffi/FFIWrapper.jl:27
- struct `RustTLogHandle` @ src/ffi/FFIWrapper.jl:31
- struct `RustTLogReaderHandle` @ src/ffi/FFIWrapper.jl:35
- struct `RustEpoch` @ src/ffi/FFIWrapper.jl:39
- struct `RustTLogSummary` @ src/ffi/FFIWrapper.jl:43
- struct `RustPageInfo` @ src/ffi/FFIWrapper.jl:49
- struct `RustSemiringPairF64` @ src/ffi/FFIWrapper.jl:58
- struct `RustSemiringPairBool` @ src/ffi/FFIWrapper.jl:63
- fn `ensure_rust_artifacts` @ src/ffi/FFIWrapper.jl:70
- fn `rust_page_read` @ src/ffi/FFIWrapper.jl:76
- fn `rust_page_epoch` @ src/ffi/FFIWrapper.jl:97
- fn `rust_page_metadata_blob` @ src/ffi/FFIWrapper.jl:104
- fn `rust_page_metadata_import` @ src/ffi/FFIWrapper.jl:118
- fn `rust_page_write_masked` @ src/ffi/FFIWrapper.jl:128
- fn `rust_delta_new` @ src/ffi/FFIWrapper.jl:140
- fn `rust_delta_free` @ src/ffi/FFIWrapper.jl:153
- fn `rust_delta_apply` @ src/ffi/FFIWrapper.jl:160
- fn `rust_allocator_new` @ src/ffi/FFIWrapper.jl:171
- fn `rust_allocator_free` @ src/ffi/FFIWrapper.jl:178
- fn `rust_allocator_allocate` @ src/ffi/FFIWrapper.jl:185
- fn `rust_allocator_release` @ src/ffi/FFIWrapper.jl:202
- fn `rust_allocator_get_page` @ src/ffi/FFIWrapper.jl:214
- fn `rust_tlog_new` @ src/ffi/FFIWrapper.jl:226
- fn `rust_tlog_free` @ src/ffi/FFIWrapper.jl:233
- fn `rust_tlog_append` @ src/ffi/FFIWrapper.jl:240
- fn `rust_tlog_reader_new` @ src/ffi/FFIWrapper.jl:247
- fn `rust_tlog_reader_free` @ src/ffi/FFIWrapper.jl:254
- fn `rust_tlog_reader_next` @ src/ffi/FFIWrapper.jl:261
- fn `rust_tlog_summary` @ src/ffi/FFIWrapper.jl:268
- fn `rust_delta_id` @ src/ffi/FFIWrapper.jl:283
- fn `rust_delta_page_id` @ src/ffi/FFIWrapper.jl:290
- fn `rust_delta_epoch` @ src/ffi/FFIWrapper.jl:297
- fn `rust_delta_is_sparse` @ src/ffi/FFIWrapper.jl:304
- fn `rust_delta_timestamp` @ src/ffi/FFIWrapper.jl:311
- fn `rust_delta_source` @ src/ffi/FFIWrapper.jl:318
- fn `rust_delta_mask` @ src/ffi/FFIWrapper.jl:334
- fn `rust_delta_payload` @ src/ffi/FFIWrapper.jl:348
- fn `rust_delta_set_intent_metadata` @ src/ffi/FFIWrapper.jl:362
- fn `rust_delta_intent_metadata` @ src/ffi/FFIWrapper.jl:379
- fn `rust_checkpoint_write` @ src/ffi/FFIWrapper.jl:393
- fn `rust_checkpoint_load` @ src/ffi/FFIWrapper.jl:402
- fn `rust_allocator_page_infos` @ src/ffi/FFIWrapper.jl:411
- fn `rust_allocator_acquire_page` @ src/ffi/FFIWrapper.jl:426
- fn `rust_get_last_error` @ src/ffi/FFIWrapper.jl:434
- fn `rust_semiring_tropical_fold_add` @ src/ffi/FFIWrapper.jl:439
- fn `rust_semiring_tropical_fold_mul` @ src/ffi/FFIWrapper.jl:450
- fn `rust_semiring_tropical_accumulate` @ src/ffi/FFIWrapper.jl:461
- fn `rust_semiring_boolean_fold_add` @ src/ffi/FFIWrapper.jl:469
- fn `rust_semiring_boolean_fold_mul` @ src/ffi/FFIWrapper.jl:480
- fn `rust_semiring_boolean_accumulate` @ src/ffi/FFIWrapper.jl:491
- struct `RustFFIError` @ src/ffi/RustErrors.jl:26
- fn `Base` @ src/ffi/RustErrors.jl:31
- fn `check_rust_error` @ src/ffi/RustErrors.jl:36
- fn `_default_message` @ src/ffi/RustErrors.jl:42
- fn `translate_error` @ src/ffi/RustErrors.jl:47

## Summary Statistics

Total elements: 841
Rust elements: 315
Julia elements: 526

By type:
- julia_fn: 439
- julia_struct: 87
- rust_fn: 220
- rust_impl: 44
- rust_struct: 51
