# Layer Dependency Report

Generated: 2025-12-17 16:38:46

## Rust Layer Graph

### Layer Order
1. `root`
2. `00_physical`
3. `01_types`
4. `01_page`
5. `02_semiring`
6. `03_dag`
7. `03_device`
8. `04_propagation`
9. `05_adaptive`
10. `06_utility`

### Layer Violations
- None detected.

### Dependency Edges
- `00_physical` → `01_page` (1 references)
  - MMSB/src/01_page/allocator.rs :: use crate :: physical :: AllocatorStats ;
- `00_physical` → `06_utility` (1 references)
  - MMSB/src/06_utility/memory_monitor.rs :: use crate :: physical :: AllocatorStats ;
- `01_page` → `03_dag` (3 references)
  - MMSB/src/03_dag/cycle_detection.rs :: use crate :: page :: PageID ;
  - MMSB/src/03_dag/shadow_graph.rs :: use crate :: page :: PageID ;
  - MMSB/src/03_dag/shadow_graph_traversal.rs :: use crate :: page :: PageID ;
- `01_page` → `04_propagation` (8 references)
  - MMSB/src/04_propagation/propagation_command_buffer.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/04_propagation/propagation_engine.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/04_propagation/propagation_fastpath.rs :: use crate :: page :: Page ;
  - MMSB/src/04_propagation/propagation_queue.rs :: use crate :: page :: { Page , PageID , PageLocation } ;
  - MMSB/src/04_propagation/throughput_engine.rs :: use crate :: page :: { ColumnarDeltaBatch , Delta , PageAllocator , PageError , PageID , merge_deltas } ;
  - MMSB/src/04_propagation/throughput_engine.rs :: use crate :: page :: { DeltaID , PageAllocator , PageID , PageLocation , Source } ;
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: page :: { Delta , DeltaID , PageAllocator , PageAllocatorConfig , PageID , PageLocation , Source } ;
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: page :: { Delta , PageError } ;
- `01_page` → `06_utility` (5 references)
  - MMSB/src/06_utility/invariant_checker.rs :: use crate :: page :: { DeviceBufferRegistry , PageAllocator , PageError , PageID } ;
  - MMSB/src/06_utility/invariant_checker.rs :: use crate :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation } ;
  - MMSB/src/06_utility/memory_monitor.rs :: use crate :: page :: PageAllocator ;
  - MMSB/src/06_utility/memory_monitor.rs :: use crate :: page :: { PageAllocator , PageAllocatorConfig , PageID , PageLocation } ;
  - MMSB/src/06_utility/provenance_tracker.rs :: use crate :: page :: PageID ;
- `01_page` → `root` (3 references)
  - MMSB/src/ffi.rs :: use crate :: page :: checkpoint ;
  - MMSB/src/ffi.rs :: use crate :: page :: tlog :: { TransactionLog , TransactionLogReader } ;
  - MMSB/src/ffi.rs :: use crate :: page :: { Delta , DeltaID , Epoch , Page , PageAllocator , PageAllocatorConfig , PageError , PageID , PageLocation , Source } ;
- `01_types` → `01_page` (14 references)
  - MMSB/src/01_page/allocator.rs :: use crate :: types :: { PageError , PageID , PageLocation } ;
  - MMSB/src/01_page/columnar_delta.rs :: use crate :: types :: Epoch ;
  - MMSB/src/01_page/columnar_delta.rs :: use crate :: types :: { DeltaID , Epoch , PageError , PageID , Source } ;
  - MMSB/src/01_page/delta.rs :: use crate :: types :: { DeltaError , DeltaID , Epoch , PageError , PageID , Source } ;
  - MMSB/src/01_page/device.rs :: use crate :: types :: PageID ;
  - MMSB/src/01_page/epoch.rs :: pub use crate :: types :: { Epoch , EpochCell } ;
  - MMSB/src/01_page/host_device_sync.rs :: use crate :: types :: PageID ;
  - MMSB/src/01_page/integrity_checker.rs :: use crate :: types :: Epoch ;
  - MMSB/src/01_page/integrity_checker.rs :: use crate :: types :: { DeltaID , Epoch , PageID } ;
  - MMSB/src/01_page/lockfree_allocator.rs :: use crate :: types :: { PageID , PageLocation } ;
  - MMSB/src/01_page/mod.rs :: pub use crate :: types :: { PageID , PageLocation , PageError , Epoch , EpochCell , DeltaID , Source , DeltaError } ;
  - MMSB/src/01_page/page.rs :: use crate :: types :: { Epoch , EpochCell , PageError , PageID , PageLocation , DeltaError } ;
  - MMSB/src/01_page/replay_validator.rs :: use crate :: types :: Epoch ;
  - MMSB/src/01_page/replay_validator.rs :: use crate :: types :: { PageError , PageID } ;
- `01_types` → `03_dag` (1 references)
  - MMSB/src/03_dag/graph_validator.rs :: use crate :: types :: PageID ;
- `01_types` → `04_propagation` (4 references)
  - MMSB/src/04_propagation/throughput_engine.rs :: use crate :: types :: DeltaError ;
  - MMSB/src/04_propagation/throughput_engine.rs :: use crate :: types :: Epoch ;
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: types :: { Epoch , GCMetrics , MemoryPressureHandler } ;
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: types :: { MemoryPressureHandler } ;
- `01_types` → `06_utility` (4 references)
  - MMSB/src/06_utility/invariant_checker.rs :: use crate :: types :: Epoch ;
  - MMSB/src/06_utility/memory_monitor.rs :: use crate :: types :: Epoch ;
  - MMSB/src/06_utility/memory_monitor.rs :: use crate :: types :: { GCMetrics , MemoryPressureHandler , PageID } ;
  - MMSB/src/06_utility/mod.rs :: pub use crate :: types :: GCMetrics ;
- `02_semiring` → `root` (1 references)
  - MMSB/src/ffi.rs :: use crate :: semiring :: { accumulate , fold_add , fold_mul , BooleanSemiring , Semiring , TropicalSemiring , } ;
- `03_dag` → `04_propagation` (2 references)
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: dag :: { EdgeType , ShadowPageGraph } ;
  - MMSB/src/04_propagation/tick_orchestrator.rs :: use crate :: dag :: { GraphValidator , ShadowPageGraph } ;
- `03_dag` → `06_utility` (4 references)
  - MMSB/src/06_utility/invariant_checker.rs :: use crate :: dag :: { EdgeType , ShadowPageGraph } ;
  - MMSB/src/06_utility/invariant_checker.rs :: use crate :: dag :: { GraphValidator , ShadowPageGraph } ;
  - MMSB/src/06_utility/provenance_tracker.rs :: use crate :: dag :: ShadowPageGraph ;
  - MMSB/src/06_utility/provenance_tracker.rs :: use crate :: dag :: { EdgeType , ShadowPageGraph } ;

### Unresolved References
- MMSB/src/ffi.rs → `use crate :: ffi_debug ;`
- MMSB/src/01_page/checkpoint.rs → `use crate :: ffi_debug ;`

## Julia Layer Graph

### Layer Order
1. `root`
2. `00_physical`
3. `01_page`
4. `01_types`
5. `02_semiring`
6. `03_dag`
7. `04_propagation`
8. `05_adaptive`
9. `06_utility`

### Layer Violations
- None detected.

### Dependency Edges
- `00_physical` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("00_physical/DeviceSync.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/GPUKernels.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/PageAllocator.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/UnifiedMemory.jl"
- `01_page` → `root` (5 references)
  - MMSB/src/MMSB.jl :: include("01_page/Delta.jl"
  - MMSB/src/MMSB.jl :: include("01_page/Page.jl"
  - MMSB/src/MMSB.jl :: include("01_page/ReplayEngine.jl"
  - MMSB/src/MMSB.jl :: include("01_page/TLog.jl"
  - MMSB/src/MMSB.jl :: using .DeltaTypes
- `01_types` → `root` (3 references)
  - MMSB/src/MMSB.jl :: include("01_types/Errors.jl"
  - MMSB/src/MMSB.jl :: include("01_types/MMSBState.jl"
  - MMSB/src/MMSB.jl :: using .MMSBStateTypes
- `02_semiring` → `root` (3 references)
  - MMSB/src/MMSB.jl :: include("02_semiring/DeltaRouter.jl"
  - MMSB/src/MMSB.jl :: include("02_semiring/Semiring.jl"
  - MMSB/src/MMSB.jl :: include("02_semiring/SemiringConfig.jl"
- `03_dag` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("03_dag/DependencyGraph.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/EventSystem.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/GraphDSL.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/ShadowPageGraph.jl"
- `04_propagation` → `root` (2 references)
  - MMSB/src/MMSB.jl :: include("04_propagation/PropagationEngine.jl"
  - MMSB/src/MMSB.jl :: include("04_propagation/PropagationScheduler.jl"
- `05_adaptive` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("05_adaptive/AdaptiveLayout.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/EntropyReduction.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/GraphRewriting.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/LocalityAnalysis.jl"
- `06_utility` → `root` (8 references)
  - MMSB/src/MMSB.jl :: include("06_utility/CostAggregation.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/ErrorRecovery.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/MemoryPressure.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/Monitoring.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/cost_functions.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/entropy_measure.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/utility_engine.jl"
  - MMSB/src/MMSB.jl :: using .Monitoring

### Unresolved References
- MMSB/benchmark/profiling/detailed_profile.jl → `using .MMSB`
- MMSB/benchmark/validate_all.jl → `using .MMSBValidationHarness`
- MMSB/benchmark/run_validation.jl → `using .MMSBBenchmarks`
- MMSB/benchmark/benchmarks.jl → `using .MMSB`
- MMSB/src/MMSB.jl → `using .API`

