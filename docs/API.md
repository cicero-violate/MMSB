# MMSB API Reference

All functions operate on declarative deltas and page-aligned state. Every entry respects the MMSB semiring constraint `state × delta → state′` (idempotent, replayable, pure). Julia snippets assume `using MMSB`.

## Configuration
```julia
state = mmsb_start(; enable_gpu=true, enable_instrumentation=false,
                   config=MMSBConfig(max_tlog_size=10_000, checkpoint_interval=60))
```

| Parameter | Description |
| --- | --- |
| `enable_gpu::Bool` | Enable CUDA allocations & kernels. Falls back to CPU if CUDA is unavailable. |
| `enable_instrumentation::Bool` | Toggles compiler/SSA hooks via InstrumentationManager. |
| `config::MMSBConfig` | Explicit configuration (logging, max log size, checkpoint cadence, default page size). |

### `MMSBConfig`
| Field | Default | Notes |
| --- | --- | --- |
| `enable_logging` | `true` | Controls TLog persistence. |
| `enable_gpu` | `true` | Mirror of keyword flag. |
| `enable_instrumentation` | `false` | Mirror of keyword flag. |
| `page_size_default::Int` | `4096` | Used by allocator fast paths. |
| `max_tlog_size::Int` | `10_000` | Threshold for log compression/checkpoints. |
| `checkpoint_interval::Int` | `60` | Seconds between background checkpoints. |

## State Lifecycle
| Function | Description |
| --- | --- |
| `mmsb_start(; kwargs...) -> MMSBState` | Builds a fresh state, registers it as the active scope, and primes GPU/instrumentation per config. |
| `mmsb_stop(state; checkpoint_path=nothing)` | Optional checkpoint + cleanup. Clears active scope when it matches `state`. |
| `@mmsb state begin ... end` | Macro that sets the implicit active state for nested helpers. Useful in REPL demos. |

## Page Management
| Function | Description |
| --- | --- |
| `create_page(state; size, location=:cpu, metadata=Dict()) -> Page` | Allocates CPU/GPU buffers, registers the page, and returns its handle. |
| `delete_page(state, page_id)` | Removes page records (defined in allocator). Ensure dependent edges are detached first. |
| `query_page(state, page_id) -> Vector{UInt8}` | Returns a copy of the page bytes for inspection/testing. |

### Locations
- `:cpu` → `CPU_LOCATION`
- `:gpu` → `GPU_LOCATION` (requires CUDA)
- `:unified` → `UNIFIED_LOCATION`

## Delta + Update Pipeline
| Function | Description |
| --- | --- |
| `update_page(state, page_id, bytes; source=:api)` | Builds a `Delta`, records it in the TLog, patches the page with masked writes, and triggers propagation. Payload length must match page size. |
| `route_delta!(state, delta)` | Lower-level entry for custom delta composition. Accepts already constructed `Delta` instances. |
| `create_delta(state, page_id, mask, bytes, source)` | Utility for building sparse/dense deltas before routing. |

### Errors
- `PageNotFoundError` — Unknown PageID.
- `InvalidDeltaError` — Payload size mismatch or invalid mask.
- `GPUMemoryError` — GPU allocation requested without CUDA support.
- `UnsupportedLocationError` — Invalid symbol passed to `location`.

## Dependency Graph + Propagation
| Function | Description |
| --- | --- |
| `add_dependency(graph, parent, child, edge_type)` | Adds an explicit dependency edge (DATA/CONTROL/GPU_SYNC/COMPILER). Throws `GraphCycleError` on cycles. |
| `remove_dependency(graph, parent, child)` | Removes edges between pages. |
| `get_children(graph, page_id)` | Returns dependents for propagation planning. |
| `get_parents(graph, page_id)` | Returns ancestors for provenance queries. |
| `PropagationEngine.propagate!(state, delta)` | Internal entry invoked by router; batches doorbells for dependents. |

## Transaction Log + Replay
| Function | Description |
| --- | --- |
| `append_to_log!(state, delta)` | Adds a delta to the log and schedules compression when thresholds are exceeded. |
| `checkpoint_log!(state, path)` | Writes checkpoint magic, version, serialized pages, and serialized deltas to disk. |
| `load_checkpoint!(state, path)` | Restores pages + TLog from a checkpoint. |
| `replay_log(state, start_epoch, end_epoch)` | Builds a detached snapshot by replaying deltas & masks between epochs. |
| `compress_log!(state)` | Merges per-page deltas to bound log growth. |
| `query_log(state; page_id, start_time, end_time, source)` | Filters deltas via metadata. |
| `get_deltas_for_page(state, page_id)` | Convenience sugar for filtering by PageID. |
| `get_deltas_in_range(state, start_idx, end_idx)` | Slice by index range. |
| `compute_log_statistics(state)` | Returns counts + source histograms for observability. |

## Instrumentation + Monitoring
| Function | Description |
| --- | --- |
| `enable_instrumentation(state)` / `disable_instrumentation(state)` | Toggle compiler hooks. |
| `get_stats()` | Returns monitoring snapshot (allocator counts, propagation latency, log size). |
| `print_stats(io=stdout)` | Human-readable summary of statistics. |
| `reset_stats!()` | Clears counters and histograms. |
| `track_delta_latency!(ns)` | Records latency metrics for delta construction. |
| `track_propagation_latency!(ns)` | Records latency metrics for propagation. |

## Usage Example
```julia
using MMSB

"""
    run_example()

Creates two pages, wires a dependency, applies a delta,
and inspects the resulting bytes/log stats.
"""
function run_example()
    state = mmsb_start()
    page_a = create_page(state; size=8, location=:cpu, metadata=Dict(:name => :source))
    page_b = create_page(state; size=8, location=:cpu, metadata=Dict(:name => :result))
    add_dependency(state.graph, page_a.id, page_b.id, DATA_DEPENDENCY)

    payload = Vector{UInt8}("mmsbproof")
    update_page(state, page_a.id, payload; source=:demo)

    @info "page_b bytes" query_page(state, page_b.id)
    @info "log stats" compute_log_statistics(state)
    mmsb_stop(state)
end

run_example()
```
