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
- struct `HostDeviceSync` @ src/00_physical/host_device_sync.rs:4
- impl `HostDeviceSync` @ src/00_physical/host_device_sync.rs:8
- fn `enqueue` @ src/00_physical/host_device_sync.rs:9
- fn `drain` @ src/00_physical/host_device_sync.rs:13

### Julia

- fn `sync_page_to_gpu` @ src/00_physical/DeviceSync.jl:36
- fn `sync_page_to_cpu` @ src/00_physical/DeviceSync.jl:67
- fn `sync_bidirectional` @ src/00_physical/DeviceSync.jl:101
- fn `ensure_page_on_device` @ src/00_physical/DeviceSync.jl:131
- fn `async_sync_page_to_gpu` @ src/00_physical/DeviceSync.jl:159
- fn `batch_sync_to_gpu` @ src/00_physical/DeviceSync.jl:190
- fn `batch_sync_to_cpu` @ src/00_physical/DeviceSync.jl:217
- fn `get_sync_statistics` @ src/00_physical/DeviceSync.jl:246
- fn `prefetch_pages_to_gpu` @ src/00_physical/DeviceSync.jl:288
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
- fn `is_unified_memory_available` @ src/00_physical/UnifiedMemory.jl:28
- fn `create_unified_page` @ src/00_physical/UnifiedMemory.jl:57
- fn `prefetch_unified_to_gpu` @ src/00_physical/UnifiedMemory.jl:75
- fn `prefetch_unified_to_cpu` @ src/00_physical/UnifiedMemory.jl:93
- fn `set_preferred_location` @ src/00_physical/UnifiedMemory.jl:115
- fn `convert_to_unified` @ src/00_physical/UnifiedMemory.jl:139
- fn `enable_read_mostly_hint` @ src/00_physical/UnifiedMemory.jl:168
- fn `disable_read_mostly_hint` @ src/00_physical/UnifiedMemory.jl:182

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
- fn `merge_deltas` @ src/01_page/delta_merge.rs:3
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
- fn `compact` @ src/01_page/tlog_compression.rs:3
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
- struct `_MetadataParser` @ src/01_page/Delta.jl:189
- fn `_decode_metadata` @ src/01_page/Delta.jl:194
- fn `_parse_metadata_value` @ src/01_page/Delta.jl:201
- fn `_parse_metadata_object` @ src/01_page/Delta.jl:224
- fn `_parse_metadata_array` @ src/01_page/Delta.jl:249
- fn `_parse_metadata_string` @ src/01_page/Delta.jl:270
- fn `_parse_metadata_number` @ src/01_page/Delta.jl:299
- fn `_skip_ws` @ src/01_page/Delta.jl:318
- fn `_consume` @ src/01_page/Delta.jl:324
- fn `_peek` @ src/01_page/Delta.jl:330
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
- fn `_with_rust_errors` @ src/01_page/TLog.jl:14
- fn `append_to_log` @ src/01_page/TLog.jl:25
- fn `log_summary` @ src/01_page/TLog.jl:34
- fn `_iterate_log` @ src/01_page/TLog.jl:42
- fn `query_log` @ src/01_page/TLog.jl:52
- fn `get_deltas_in_range` @ src/01_page/TLog.jl:83
- fn `compute_log_statistics` @ src/01_page/TLog.jl:91
- fn `replay_log` @ src/01_page/TLog.jl:100
- fn `checkpoint_log` @ src/01_page/TLog.jl:106
- fn `load_checkpoint` @ src/01_page/TLog.jl:114
- fn `_refresh_pages` @ src/01_page/TLog.jl:123

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
- struct `SemiringOps` @ src/02_semiring/Semiring.jl:5
- fn `tropical_semiring` @ src/02_semiring/Semiring.jl:12
- fn `boolean_semiring` @ src/02_semiring/Semiring.jl:21
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
- struct `PropagationQueue` @ src/04_propagation/propagation_queue.rs:6
- impl `PropagationQueue` @ src/04_propagation/propagation_queue.rs:10
- fn `push` @ src/04_propagation/propagation_queue.rs:11
- fn `pop` @ src/04_propagation/propagation_queue.rs:15
- fn `enqueue_sparse` @ src/04_propagation/sparse_message_passing.rs:5

### Julia

- struct `PropagationQueue` @ src/04_propagation/PropagationEngine.jl:31
- fn `_buffer` @ src/04_propagation/PropagationEngine.jl:41
- fn `register_recompute_fn` @ src/04_propagation/PropagationEngine.jl:53
- fn `register_passthrough_recompute` @ src/04_propagation/PropagationEngine.jl:66
- fn `queue_recomputation` @ src/04_propagation/PropagationEngine.jl:80
- fn `propagate_change` @ src/04_propagation/PropagationEngine.jl:93
- fn `propagate_change` @ src/04_propagation/PropagationEngine.jl:98
- fn `_aggregate_children` @ src/04_propagation/PropagationEngine.jl:108
- fn `_execute_command_buffer` @ src/04_propagation/PropagationEngine.jl:121
- fn `_apply_edges` @ src/04_propagation/PropagationEngine.jl:129
- fn `_handle_data_dependency` @ src/04_propagation/PropagationEngine.jl:142
- fn `_collect_descendants` @ src/04_propagation/PropagationEngine.jl:155
- fn `schedule_propagation` @ src/04_propagation/PropagationEngine.jl:175
- fn `execute_propagation` @ src/04_propagation/PropagationEngine.jl:191
- fn `recompute_page` @ src/04_propagation/PropagationEngine.jl:205
- fn `mark_page_stale` @ src/04_propagation/PropagationEngine.jl:229
- fn `schedule_gpu_sync` @ src/04_propagation/PropagationEngine.jl:240
- fn `invalidate_compilation` @ src/04_propagation/PropagationEngine.jl:250
- fn `topological_order_subset` @ src/04_propagation/PropagationEngine.jl:263
- fn `schedule` @ src/04_propagation/PropagationScheduler.jl:8

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

## 07_intention


### Julia

- struct `Intention` @ src/07_intention/IntentionTypes.jl:17
- struct `Goal` @ src/07_intention/IntentionTypes.jl:31
- struct `IntentionState` @ src/07_intention/IntentionTypes.jl:44
- fn `IntentionState` @ src/07_intention/IntentionTypes.jl:51
- struct `DeltaSpec` @ src/07_intention/UpsertPlan.jl:5
- struct `UpsertPlan` @ src/07_intention/UpsertPlan.jl:11
- fn `validate_plan` @ src/07_intention/UpsertPlan.jl:18
- struct `AttractorField` @ src/07_intention/attractor_states.jl:15
- fn `compute_gradient` @ src/07_intention/attractor_states.jl:25
- fn `evolve_state` @ src/07_intention/attractor_states.jl:44
- fn `find_nearest_attractor` @ src/07_intention/attractor_states.jl:54
- fn `utility_gradient` @ src/07_intention/goal_emergence.jl:17
- fn `detect_goals` @ src/07_intention/goal_emergence.jl:38
- fn `mask_to_bytes` @ src/07_intention/intent_lowering.jl:17
- fn `lower_intent_to_deltaspec` @ src/07_intention/intent_lowering.jl:21
- fn `execute_upsert_plan` @ src/07_intention/intent_lowering.jl:29
- fn `form_intention` @ src/07_intention/intention_engine.jl:18
- fn `evaluate_intention` @ src/07_intention/intention_engine.jl:50
- fn `select_best_intention` @ src/07_intention/intention_engine.jl:66
- struct `Preference` @ src/07_intention/structural_preferences.jl:15
- fn `evaluate_preference` @ src/07_intention/structural_preferences.jl:26
- fn `apply_preferences` @ src/07_intention/structural_preferences.jl:40

## 08_reasoning


### Julia

- struct `Constraint` @ src/08_reasoning/ReasoningTypes.jl:19
- struct `Dependency` @ src/08_reasoning/ReasoningTypes.jl:33
- struct `Pattern` @ src/08_reasoning/ReasoningTypes.jl:41
- struct `PatternMatch` @ src/08_reasoning/ReasoningTypes.jl:49
- struct `Rule` @ src/08_reasoning/ReasoningTypes.jl:62
- struct `Inference` @ src/08_reasoning/ReasoningTypes.jl:70
- struct `InferenceResult` @ src/08_reasoning/ReasoningTypes.jl:77
- struct `ReasoningState` @ src/08_reasoning/ReasoningTypes.jl:84
- fn `ReasoningState` @ src/08_reasoning/ReasoningTypes.jl:92
- fn `propagate_constraints` @ src/08_reasoning/constraint_propagation.jl:17
- fn `forward_propagate` @ src/08_reasoning/constraint_propagation.jl:61
- fn `backward_propagate` @ src/08_reasoning/constraint_propagation.jl:94
- fn `infer_dependencies` @ src/08_reasoning/dependency_inference.jl:17
- fn `analyze_edge_type` @ src/08_reasoning/dependency_inference.jl:44
- fn `compute_dependency_strength` @ src/08_reasoning/dependency_inference.jl:70
- fn `count_paths` @ src/08_reasoning/dependency_inference.jl:88
- fn `analyze_flow` @ src/08_reasoning/dependency_inference.jl:112
- fn `compute_critical_path` @ src/08_reasoning/dependency_inference.jl:139
- fn `deduce` @ src/08_reasoning/logic_engine.jl:17
- fn `abduce` @ src/08_reasoning/logic_engine.jl:43
- fn `induce` @ src/08_reasoning/logic_engine.jl:68
- fn `unify_constraints` @ src/08_reasoning/logic_engine.jl:104
- fn `find_patterns` @ src/08_reasoning/pattern_formation.jl:17
- fn `extract_subgraphs` @ src/08_reasoning/pattern_formation.jl:58
- fn `grow_subgraph` @ src/08_reasoning/pattern_formation.jl:71
- fn `extract_subgraph_signature` @ src/08_reasoning/pattern_formation.jl:97
- fn `extract_edges` @ src/08_reasoning/pattern_formation.jl:127
- fn `match_pattern` @ src/08_reasoning/pattern_formation.jl:148
- fn `initialize_reasoning` @ src/08_reasoning/reasoning_engine.jl:23
- fn `reason_over_dag` @ src/08_reasoning/reasoning_engine.jl:51
- fn `perform_inference` @ src/08_reasoning/reasoning_engine.jl:102
- fn `evaluate_rules` @ src/08_reasoning/rule_evaluation.jl:17
- fn `apply_rule` @ src/08_reasoning/rule_evaluation.jl:47
- fn `create_default_rules` @ src/08_reasoning/rule_evaluation.jl:59
- fn `infer_from_structure` @ src/08_reasoning/structural_inference.jl:17
- fn `derive_constraints` @ src/08_reasoning/structural_inference.jl:58
- fn `check_consistency` @ src/08_reasoning/structural_inference.jl:76

## 09_planning


### Julia

- struct `State` @ src/09_planning/PlanningTypes.jl:12
- struct `Action` @ src/09_planning/PlanningTypes.jl:18
- struct `Goal` @ src/09_planning/PlanningTypes.jl:26
- struct `Plan` @ src/09_planning/PlanningTypes.jl:34
- struct `SearchNode` @ src/09_planning/PlanningTypes.jl:43
- struct `Strategy` @ src/09_planning/PlanningTypes.jl:52
- struct `RolloutResult` @ src/09_planning/PlanningTypes.jl:59
- struct `DecisionGraph` @ src/09_planning/PlanningTypes.jl:66
- struct `PlanningState` @ src/09_planning/PlanningTypes.jl:72
- struct `PlanMetrics` @ src/09_planning/PlanningTypes.jl:81
- fn `PlanningState` @ src/09_planning/PlanningTypes.jl:88
- fn `build_decision_graph` @ src/09_planning/decision_graphs.jl:17
- fn `expand_graph` @ src/09_planning/decision_graphs.jl:31
- fn `apply_action_simple` @ src/09_planning/decision_graphs.jl:52
- fn `find_optimal_path` @ src/09_planning/decision_graphs.jl:63
- fn `prune_graph` @ src/09_planning/decision_graphs.jl:115
- fn `decompose_goal` @ src/09_planning/goal_decomposition.jl:17
- fn `create_subgoal_hierarchy` @ src/09_planning/goal_decomposition.jl:59
- fn `order_subgoals` @ src/09_planning/goal_decomposition.jl:76
- fn `score_subgoal` @ src/09_planning/goal_decomposition.jl:83
- fn `estimate_achievability` @ src/09_planning/goal_decomposition.jl:89
- fn `optimize_plan` @ src/09_planning/optimization_planning.jl:17
- fn `extract_parameters` @ src/09_planning/optimization_planning.jl:35
- fn `compute_gradient` @ src/09_planning/optimization_planning.jl:44
- fn `reconstruct_plan` @ src/09_planning/optimization_planning.jl:60
- fn `gradient_descent_planning` @ src/09_planning/optimization_planning.jl:88
- fn `evaluate_action_sequence` @ src/09_planning/optimization_planning.jl:107
- fn `compute_sequence_gradient` @ src/09_planning/optimization_planning.jl:120
- fn `actions_from_params` @ src/09_planning/optimization_planning.jl:138
- fn `prepare_for_enzyme` @ src/09_planning/optimization_planning.jl:158
- fn `create_plan` @ src/09_planning/planning_engine.jl:24
- fn `execute_planning` @ src/09_planning/planning_engine.jl:49
- fn `replan` @ src/09_planning/planning_engine.jl:96
- fn `value_iteration` @ src/09_planning/rl_planning.jl:17
- fn `immediate_reward` @ src/09_planning/rl_planning.jl:50
- fn `expected_next_value` @ src/09_planning/rl_planning.jl:54
- fn `policy_iteration` @ src/09_planning/rl_planning.jl:66
- fn `evaluate_policy` @ src/09_planning/rl_planning.jl:116
- fn `q_learning` @ src/09_planning/rl_planning.jl:131
- fn `temporal_difference` @ src/09_planning/rl_planning.jl:155
- fn `simulate_plan` @ src/09_planning/rollout_simulation.jl:18
- fn `parallel_rollout` @ src/09_planning/rollout_simulation.jl:43
- fn `evaluate_outcome` @ src/09_planning/rollout_simulation.jl:63
- fn `astar_search` @ src/09_planning/search_algorithms.jl:17
- fn `mcts_search` @ src/09_planning/search_algorithms.jl:66
- struct `MCTSNode` @ src/09_planning/search_algorithms.jl:89
- fn `select_node` @ src/09_planning/search_algorithms.jl:100
- fn `best_uct_child` @ src/09_planning/search_algorithms.jl:107
- fn `expand_node` @ src/09_planning/search_algorithms.jl:129
- fn `simulate` @ src/09_planning/search_algorithms.jl:143
- fn `backpropagate` @ src/09_planning/search_algorithms.jl:165
- fn `compute_heuristic` @ src/09_planning/search_algorithms.jl:179
- fn `can_apply` @ src/09_planning/search_algorithms.jl:187
- fn `apply_action` @ src/09_planning/search_algorithms.jl:191
- fn `is_terminal` @ src/09_planning/search_algorithms.jl:205
- fn `reconstruct_plan` @ src/09_planning/search_algorithms.jl:209
- fn `extract_plan_from_mcts` @ src/09_planning/search_algorithms.jl:224
- fn `generate_strategies` @ src/09_planning/strategy_generation.jl:19
- fn `hierarchical_planning` @ src/09_planning/strategy_generation.jl:49
- fn `select_strategy` @ src/09_planning/strategy_generation.jl:75
- fn `adapt_strategy` @ src/09_planning/strategy_generation.jl:108

## 10_agent_interface


### Julia

- struct `AgentAction` @ src/10_agent_interface/AgentProtocol.jl:14
- fn `observe` @ src/10_agent_interface/AgentProtocol.jl:25
- fn `act` @ src/10_agent_interface/AgentProtocol.jl:32
- fn `plan` @ src/10_agent_interface/AgentProtocol.jl:39
- fn `enable_base_hooks` @ src/10_agent_interface/BaseHook.jl:37
- fn `invoke` @ src/10_agent_interface/BaseHook.jl:55
- fn `disable_base_hooks` @ src/10_agent_interface/BaseHook.jl:74
- fn `hook_invoke` @ src/10_agent_interface/BaseHook.jl:107
- fn `hook_setfield` @ src/10_agent_interface/BaseHook.jl:135
- fn `hook_getfield` @ src/10_agent_interface/BaseHook.jl:160
- struct `MMSBInterpreter` @ src/10_agent_interface/CompilerHooks.jl:37
- fn `MMSBInterpreter` @ src/10_agent_interface/CompilerHooks.jl:42
- fn `Core` @ src/10_agent_interface/CompilerHooks.jl:84
- fn `Core` @ src/10_agent_interface/CompilerHooks.jl:117
- fn `Core` @ src/10_agent_interface/CompilerHooks.jl:159
- fn `log_inference_start` @ src/10_agent_interface/CompilerHooks.jl:194
- fn `log_inference_result` @ src/10_agent_interface/CompilerHooks.jl:212
- fn `create_inference_pages` @ src/10_agent_interface/CompilerHooks.jl:235
- fn `enable_compiler_hooks` @ src/10_agent_interface/CompilerHooks.jl:259
- fn `disable_compiler_hooks` @ src/10_agent_interface/CompilerHooks.jl:273
- fn `enable_core_hooks` @ src/10_agent_interface/CoreHooks.jl:32
- fn `disable_core_hooks` @ src/10_agent_interface/CoreHooks.jl:49
- fn `hook_codeinfo_creation` @ src/10_agent_interface/CoreHooks.jl:72
- fn `hook_methodinstance` @ src/10_agent_interface/CoreHooks.jl:103
- struct `InstrumentationConfig` @ src/10_agent_interface/InstrumentationManager.jl:23
- fn `InstrumentationConfig` @ src/10_agent_interface/InstrumentationManager.jl:33
- fn `enable_instrumentation` @ src/10_agent_interface/InstrumentationManager.jl:51
- fn `disable_instrumentation` @ src/10_agent_interface/InstrumentationManager.jl:77
- fn `configure_instrumentation` @ src/10_agent_interface/InstrumentationManager.jl:100
- fn `create_checkpoint` @ src/10_agent_interface/checkpoint_api.jl:11
- fn `restore_checkpoint` @ src/10_agent_interface/checkpoint_api.jl:17
- fn `list_checkpoints` @ src/10_agent_interface/checkpoint_api.jl:21
- struct `Subscription` @ src/10_agent_interface/event_subscription.jl:19
- fn `subscribe_to_events` @ src/10_agent_interface/event_subscription.jl:29
- fn `unsubscribe` @ src/10_agent_interface/event_subscription.jl:36
- fn `emit_event` @ src/10_agent_interface/event_subscription.jl:40

## 11_agents


### Julia

- struct `AgentState` @ src/11_agents/AgentTypes.jl:12
- struct `AgentMemory` @ src/11_agents/AgentTypes.jl:20
- fn `push_memory` @ src/11_agents/AgentTypes.jl:29
- fn `gradient_descent_step` @ src/11_agents/enzyme_integration.jl:11
- fn `autodiff_loss` @ src/11_agents/enzyme_integration.jl:18
- struct `HybridAgent` @ src/11_agents/hybrid_agent.jl:15
- fn `observe` @ src/11_agents/hybrid_agent.jl:23
- fn `symbolic_step` @ src/11_agents/hybrid_agent.jl:30
- fn `neural_step` @ src/11_agents/hybrid_agent.jl:36
- fn `create_value_network` @ src/11_agents/lux_models.jl:11
- fn `create_policy_network` @ src/11_agents/lux_models.jl:17
- struct `PlanningAgent` @ src/11_agents/planning_agent.jl:15
- fn `observe` @ src/11_agents/planning_agent.jl:23
- fn `generate_plan` @ src/11_agents/planning_agent.jl:30
- fn `execute_plan_step` @ src/11_agents/planning_agent.jl:36
- struct `RLAgent` @ src/11_agents/rl_agent.jl:13
- fn `RLAgent` @ src/11_agents/rl_agent.jl:20
- fn `observe` @ src/11_agents/rl_agent.jl:24
- fn `compute_reward` @ src/11_agents/rl_agent.jl:32
- fn `train_step` @ src/11_agents/rl_agent.jl:37
- struct `Rule` @ src/11_agents/symbolic_agent.jl:14
- struct `SymbolicAgent` @ src/11_agents/symbolic_agent.jl:20
- fn `observe` @ src/11_agents/symbolic_agent.jl:27
- fn `infer_rules` @ src/11_agents/symbolic_agent.jl:32
- fn `apply_rule` @ src/11_agents/symbolic_agent.jl:37

## 12_applications


### Julia

- struct `Asset` @ src/12_applications/financial_modeling.jl:11
- struct `Portfolio` @ src/12_applications/financial_modeling.jl:17
- fn `compute_value` @ src/12_applications/financial_modeling.jl:25
- fn `rebalance` @ src/12_applications/financial_modeling.jl:33
- struct `MMSBContext` @ src/12_applications/llm_tools.jl:11
- fn `query_llm` @ src/12_applications/llm_tools.jl:19
- fn `store_llm_response` @ src/12_applications/llm_tools.jl:25
- struct `ReasoningContext` @ src/12_applications/memory_driven_reasoning.jl:12
- fn `reason_over_memory` @ src/12_applications/memory_driven_reasoning.jl:18
- fn `temporal_reasoning` @ src/12_applications/memory_driven_reasoning.jl:23
- struct `AgentCoordinator` @ src/12_applications/multi_agent_system.jl:11
- fn `register_agent` @ src/12_applications/multi_agent_system.jl:19
- fn `coordinate_step` @ src/12_applications/multi_agent_system.jl:23
- struct `Entity` @ src/12_applications/world_simulation.jl:13
- struct `World` @ src/12_applications/world_simulation.jl:19
- fn `add_entity` @ src/12_applications/world_simulation.jl:28
- fn `simulate_step` @ src/12_applications/world_simulation.jl:35

## root

### Rust

- fn `main` @ build.rs:1
- struct `PageHandle` @ src/ffi.rs:15
- impl `PageHandle` @ src/ffi.rs:19
- fn `null` @ src/ffi.rs:20
- struct `DeltaHandle` @ src/ffi.rs:29
- impl `DeltaHandle` @ src/ffi.rs:33
- fn `null` @ src/ffi.rs:34
- struct `AllocatorHandle` @ src/ffi.rs:43
- impl `AllocatorHandle` @ src/ffi.rs:47
- fn `null` @ src/ffi.rs:48
- struct `TLogHandle` @ src/ffi.rs:57
- impl `TLogHandle` @ src/ffi.rs:61
- fn `null` @ src/ffi.rs:62
- struct `TLogReaderHandle` @ src/ffi.rs:71
- impl `TLogReaderHandle` @ src/ffi.rs:75
- fn `null` @ src/ffi.rs:76
- fn `set_last_error` @ src/ffi.rs:99
- fn `log_error_code` @ src/ffi.rs:105
- struct `TLogSummary` @ src/ffi.rs:125
- struct `PageInfoABI` @ src/ffi.rs:133
- struct `EpochABI` @ src/ffi.rs:144
- impl `From` @ src/ffi.rs:148
- fn `from` @ src/ffi.rs:149
- fn `convert_location` @ src/ffi.rs:154
- fn `mask_from_bytes` @ src/ffi.rs:158
- fn `vec_from_ptr` @ src/ffi.rs:185
- fn `dense_delta` @ tests/delta_validation.rs:3
- fn `validates_dense_lengths` @ tests/delta_validation.rs:18
- fn `rejects_mismatched_dense_lengths` @ tests/delta_validation.rs:24
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

### Julia

- fn `_start_state` @ benchmark/benchmarks.jl:37
- fn `_stop_state` @ benchmark/benchmarks.jl:41
- fn `_page` @ benchmark/benchmarks.jl:47
- fn `_populate_pages` @ benchmark/benchmarks.jl:51
- fn `_seed_pages` @ benchmark/benchmarks.jl:55
- fn `_replay_sequence` @ benchmark/benchmarks.jl:62
- fn `_stress_updates` @ benchmark/benchmarks.jl:68
- fn `_link_chain` @ benchmark/benchmarks.jl:75
- fn `_checkpoint` @ benchmark/benchmarks.jl:83
- fn `_trial_to_dict` @ benchmark/benchmarks.jl:251
- fn `_select_suite` @ benchmark/benchmarks.jl:262
- fn `_to_mutable` @ benchmark/benchmarks.jl:274
- fn `run_benchmarks` @ benchmark/benchmarks.jl:284
- fn `compare_with_baseline` @ benchmark/benchmarks.jl:319
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
- fn `ensure_rust_artifacts` @ src/ffi/FFIWrapper.jl:60
- fn `rust_page_read` @ src/ffi/FFIWrapper.jl:66
- fn `rust_page_epoch` @ src/ffi/FFIWrapper.jl:87
- fn `rust_page_metadata_blob` @ src/ffi/FFIWrapper.jl:94
- fn `rust_page_metadata_import` @ src/ffi/FFIWrapper.jl:108
- fn `rust_page_write_masked` @ src/ffi/FFIWrapper.jl:118
- fn `rust_delta_new` @ src/ffi/FFIWrapper.jl:130
- fn `rust_delta_free` @ src/ffi/FFIWrapper.jl:143
- fn `rust_delta_apply` @ src/ffi/FFIWrapper.jl:150
- fn `rust_allocator_new` @ src/ffi/FFIWrapper.jl:161
- fn `rust_allocator_free` @ src/ffi/FFIWrapper.jl:168
- fn `rust_allocator_allocate` @ src/ffi/FFIWrapper.jl:175
- fn `rust_allocator_release` @ src/ffi/FFIWrapper.jl:192
- fn `rust_allocator_get_page` @ src/ffi/FFIWrapper.jl:204
- fn `rust_tlog_new` @ src/ffi/FFIWrapper.jl:216
- fn `rust_tlog_free` @ src/ffi/FFIWrapper.jl:223
- fn `rust_tlog_append` @ src/ffi/FFIWrapper.jl:230
- fn `rust_tlog_reader_new` @ src/ffi/FFIWrapper.jl:237
- fn `rust_tlog_reader_free` @ src/ffi/FFIWrapper.jl:244
- fn `rust_tlog_reader_next` @ src/ffi/FFIWrapper.jl:251
- fn `rust_tlog_summary` @ src/ffi/FFIWrapper.jl:258
- fn `rust_delta_id` @ src/ffi/FFIWrapper.jl:273
- fn `rust_delta_page_id` @ src/ffi/FFIWrapper.jl:280
- fn `rust_delta_epoch` @ src/ffi/FFIWrapper.jl:287
- fn `rust_delta_is_sparse` @ src/ffi/FFIWrapper.jl:294
- fn `rust_delta_timestamp` @ src/ffi/FFIWrapper.jl:301
- fn `rust_delta_source` @ src/ffi/FFIWrapper.jl:308
- fn `rust_delta_mask` @ src/ffi/FFIWrapper.jl:324
- fn `rust_delta_payload` @ src/ffi/FFIWrapper.jl:338
- fn `rust_delta_set_intent_metadata` @ src/ffi/FFIWrapper.jl:352
- fn `rust_delta_intent_metadata` @ src/ffi/FFIWrapper.jl:369
- fn `rust_checkpoint_write` @ src/ffi/FFIWrapper.jl:383
- fn `rust_checkpoint_load` @ src/ffi/FFIWrapper.jl:392
- fn `rust_allocator_page_infos` @ src/ffi/FFIWrapper.jl:401
- fn `rust_allocator_acquire_page` @ src/ffi/FFIWrapper.jl:416
- fn `rust_get_last_error` @ src/ffi/FFIWrapper.jl:424
- struct `RustFFIError` @ src/ffi/RustErrors.jl:26
- fn `Base` @ src/ffi/RustErrors.jl:31
- fn `check_rust_error` @ src/ffi/RustErrors.jl:36
- fn `_default_message` @ src/ffi/RustErrors.jl:42
- fn `translate_error` @ src/ffi/RustErrors.jl:47

## Summary Statistics

Total elements: 730
Rust elements: 248
Julia elements: 482

By type:
- julia_fn: 404
- julia_struct: 78
- rust_fn: 170
- rust_impl: 36
- rust_struct: 42
