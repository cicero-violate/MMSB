# MMSB Architecture

MMSB (Memory-Mapped State Bus) is a semiring-governed state fabric where every computation is expressed as `state ⊕ Δ → state′` and `state′ ⊗ dependency → propagated_state`. The substrate exposes page-aligned memory forms, deterministic deltas, and a declarative dependency graph so CPU, GPU, and compiler layers share a single canonical truth.

## Layered Stack
| Layer | Description |
| --- | --- |
| **01_types** | Defines page, delta, graph, error, and state structs. All structs are page-aligned and immutable outside MMSB mutations. |
| **02_runtime** | Implements the allocator, delta router, TLog, and replay engine. Deltas never mutate pages directly; they become masked writes scheduled by the router. |
| **03_device** | Contains CUDA kernels, device sync helpers, and unified memory shims. GPU buffers stay coherent with CPU mirrors through explicit doorbells. |
| **04_instrumentation** | Hooks Julia’s compiler pipelines (Base/Core) into MMSB via ShadowPageGraph edges so SSA/IR artifacts live on pages. |
| **05_graph** | EventSystem plus PropagationEngine. Maintains the ShadowPageGraph, emits propagation batches, and guarantees deterministic recomputation order. |
| **utils** | Monitoring counters and telemetry exposed through `get_stats`, `track_*` helpers, and baseline benchmarks. |

## Control Flow Walkthrough
1. **State initialization** — `mmsb_start` builds `MMSBState`, the ShadowPageGraph, device managers, and sets configuration flags (logging, GPU, instrumentation).
2. **Page lifecycle** — `create_page` allocates CPU/GPU buffers, masks, and metadata. All allocations are page-sized and registered in the state registry.
3. **Delta creation** — Clients prepare byte payloads. `update_page` resolves masks, constructs `Delta` instances (possibly sparse), and hands them to `DeltaRouter.route_delta!`.
4. **Routing & propagation** — DeltaRouter enqueues masked writes, updates page epochs, records the delta into the TLog, and raises propagation events.
5. **Propagation** — PropagationEngine queries ShadowPageGraph to find dependent pages and composes transformations in topological order. Branchless command buffers define how each dependency reacts.
6. **Logging/Replay** — Every delta is serialized into the TLog. `checkpoint_log!` snapshots the pages + log, while `replay_log` rebuilds any epoch slice. Compression merges deltas to keep the log bounded.
7. **Instrumentation** — Compiler hooks emit SSA/IR forms as pages and connect them to their source data via ShadowPageGraph edges, enabling deterministic re-analysis.
8. **Monitoring** — `track_delta_latency!`/`track_propagation_latency!` record latency histograms, allocator pressure, and log sizes so runtimes can enforce budgets.

## ShadowPageGraph & Propagation
- **Edge semantics:** Distinguish data, control, GPU sync, and compiler dependencies. Each edge is explicit; there are no hidden globals or implied parents.
- **Lawful composition:** PropagationEngine batches masked writes according to a topological order to avoid locks during the actual compute pass.
- **Event system:** The branchless doorbell emitter notifies each subscription channel (CPU propagators, GPU synchronizer, instrumentation callbacks) with tuple payloads. Consumers apply declarative masks (⊗) rather than imperative logic.

## Device Synchronization
- CUDA pages live beside CPU pages but always share the same PageID.
- DeviceSync ensures GPU kernels see coherent bytes by:
  1. Issuing command buffers describing host→device or device→host intentions.
  2. Scheduling CUDA streams that read/write unified ring buffers.
  3. Updating the transaction log once the doorbell confirms completion.
- Unified memory pages opt into a lazy faulting path while still obeying the same delta semantics.

## Instrumentation Hooks
- Base and Core hooks capture world-age and SSA deltas as MMSB pages.
- CompilerHooks maintains a ShadowPageGraph subgraph linking source modules → lowered code → typed IR → optimized IR.
- InstrumentationManager exposes toggles (`enable_instrumentation`, `disable_instrumentation`) so profiling or deterministic replay is opt-in per state.

## Monitoring & Benchmarks
- Monitoring.jl exposes counters via `get_stats`/`print_stats` for allocators, propagation, and TLog.
- Benchmarks (`benchmark/benchmarks.jl`) reuse the command buffer architecture to exercise allocations, propagation chains, and replay loops. The baseline JSON becomes part of the release artifact.

## Release & Future Work
- **Complete:** Core runtime, GPU runtime, graph engine, public API, monitoring, benchmark suite, and now documentation.
- **Deferred:** Propagation micro-optimizations (lock contention, batching), sparse delta SIMD acceleration, and allocator fast paths. These remain Phase 2 items post-release.
