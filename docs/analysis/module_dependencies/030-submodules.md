# Submodules

## MMSB/src/00_physical/mod.rs (00_physical)

Module `mod`

- `allocator_stats`
- `gpu_memory_pool`
- `nccl_integration`

## MMSB/src/01_page/allocator.rs (01_page)

Module `allocator`

- `tests`

## MMSB/src/01_page/columnar_delta.rs (01_page)

Module `columnar_delta`

- `tests`

## MMSB/src/01_page/integrity_checker.rs (01_page)

Module `integrity_checker`

- `tests`

## MMSB/src/01_page/mod.rs (01_page)

Module `mod`

- `allocator`
- `checkpoint`
- `columnar_delta`
- `delta`
- `delta_merge`
- `delta_validation`
- `device`
- `device_registry`
- `epoch`
- `host_device_sync`
- `integrity_checker`
- `lockfree_allocator`
- `page`
- `replay_validator`
- `simd_mask`
- `tlog`
- `tlog_compression`
- `tlog_replay`
- `tlog_serialization`

## MMSB/src/01_page/replay_validator.rs (01_page)

Module `replay_validator`

- `tests`

## MMSB/src/01_types/mod.rs (01_types)

Module `mod`

- `delta_types`
- `epoch_types`
- `gc`
- `page_types`

## MMSB/src/02_semiring/mod.rs (02_semiring)

Module `mod`

- `purity_validator`
- `semiring_ops`
- `semiring_types`
- `standard_semirings`

## MMSB/src/02_semiring/purity_validator.rs (02_semiring)

Module `purity_validator`

- `tests`

## MMSB/src/03_dag/graph_validator.rs (03_dag)

Module `graph_validator`

- `tests`

## MMSB/src/03_dag/mod.rs (03_dag)

Module `mod`

- `cycle_detection`
- `edge_types`
- `graph_validator`
- `shadow_graph`
- `shadow_graph_mod`
- `shadow_graph_traversal`

## MMSB/src/03_device/mod.rs (03_device)

Module `mod`

- `device`
- `device_registry`
- `host_device_sync`

## MMSB/src/04_propagation/mod.rs (04_propagation)

Module `mod`

- `propagation_command_buffer`
- `propagation_engine`
- `propagation_fastpath`
- `propagation_queue`
- `ring_buffer`
- `sparse_message_passing`
- `throughput_engine`
- `tick_orchestrator`

## MMSB/src/04_propagation/propagation_queue.rs (04_propagation)

Module `propagation_queue`

- `tests`

## MMSB/src/04_propagation/ring_buffer.rs (04_propagation)

Module `ring_buffer`

- `tests`

## MMSB/src/04_propagation/throughput_engine.rs (04_propagation)

Module `throughput_engine`

- `tests`

## MMSB/src/04_propagation/tick_orchestrator.rs (04_propagation)

Module `tick_orchestrator`

- `tests`

## MMSB/src/05_adaptive/locality_optimizer.rs (05_adaptive)

Module `locality_optimizer`

- `tests`

## MMSB/src/05_adaptive/memory_layout.rs (05_adaptive)

Module `memory_layout`

- `tests`

## MMSB/src/05_adaptive/mod.rs (05_adaptive)

Module `mod`

- `locality_optimizer`
- `memory_layout`
- `page_clustering`

## MMSB/src/05_adaptive/page_clustering.rs (05_adaptive)

Module `page_clustering`

- `tests`

## MMSB/src/06_utility/invariant_checker.rs (06_utility)

Module `invariant_checker`

- `tests`

## MMSB/src/06_utility/memory_monitor.rs (06_utility)

Module `memory_monitor`

- `tests`

## MMSB/src/06_utility/mod.rs (06_utility)

Module `mod`

- `cpu_features`
- `invariant_checker`
- `memory_monitor`
- `provenance_tracker`
- `telemetry`

## MMSB/src/06_utility/provenance_tracker.rs (06_utility)

Module `provenance_tracker`

- `tests`

## MMSB/src/06_utility/telemetry.rs (06_utility)

Module `telemetry`

- `tests`

## MMSB/src/lib.rs (root)

Module `lib`

- `adaptive`
- `dag`
- `ffi`
- `logging`
- `page`
- `physical`
- `propagation`
- `semiring`
- `types`
- `utility`

