# MMSB — Memory-Mapped State Bus

Self-optimizing GPU-accelerated memory system with autonomous reasoning and planning. MMSB is a deterministic, delta-driven shared-memory fabric that lets CPU, GPU, and compiler subsystems share page-aligned state under a single transaction log. Every operation follows the MMSB semiring law `state × delta → state′`, enabling deterministic replay and algebraic propagation.



## Why MMSB
- **Deterministic replay** — Byte-level deltas, epochs, and transaction log (TLog) enable checkpoint/reconstruction
- **CPU/GPU coherence** — Unified API across CPU pages, CUDA buffers, and unified memory
- **Declarative graph** — ShadowPageGraph captures dependencies; propagation is algebraic
- **Self-optimization** — Adaptive memory layout, graph rewriting, entropy reduction
- **Autonomous reasoning** — Provided by [`MMSB-top`](../MMSB-top) (Layers 7-8: structural inference, goal emergence)
- **Planning & agents** — Provided by [`MMSB-top`](../MMSB-top) (Layers 9-11: planners, RL, symbolic/hybrid agents)
- **Instrumentation** — Compiler hooks and agent interface moved to [`MMSB-top`](../MMSB-top)
- **Observability** — Built-in monitoring: allocator pressure, delta latency, propagation metrics

## Architecture (Core Layers 0-6)

```
Layer 6:  Utility Engine   → Cost Functions, Telemetry, Entropy Measurement
Layer 5:  Adaptive Memory  → Layout Optimization, Clustering, Graph Rewriting
Layer 4:  Propagation      → CPU/GPU Message Passing, Sparse Propagation
Layer 3:  DAG/Graph        → ShadowPageGraph, Dependency Tracking, Cycles
Layer 2:  Semiring Algebra → Delta Router, Merge Operations, Tropical/Boolean
Layer 1:  Page Layer       → Pages, Deltas, TLog, Checkpoint/Replay
Layer 0:  Physical Memory  → Page Allocator, Unified Memory, GPU Kernels
```

 

## Quick Start

### Build
```bash
cargo build --release
```

### Test
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Basic Usage
```julia
using MMSB

# Create state with GPU support
state = mmsb_start(enable_gpu=true)

# Allocate page
page = create_page(state, size=1024, location=:gpu)

# Apply delta
update_page(state, page.id, rand(UInt8, 1024))

# Read data
data = query_page(state, page.id)

# Checkpoint and shutdown
mmsb_stop(state, checkpoint_path="state.ckpt")
```

### Advanced: Cognitive Stack (via MMSB-top)
```julia
using MMSB
using MMSBTop

state = mmsb_start()

# Hook Layer 6 utility into Layer 7 intention engine
utility = MMSB.UtilityEngine.UtilityState()
layout_stub = (placement = Dict{UInt64, Any}(), locality_score = 0.0)
intention = MMSBTop.IntentionEngine.form_intention(utility, layout_stub, 1)

# Use application helpers (Layer 12)
ctx = MMSBTop.LLMTools.MMSBContext(state)
response = MMSBTop.LLMTools.query_llm(ctx, "Summarize allocator health")
```

## Repository Map

| Path                      | Purpose                                     |
|---------------------------+---------------------------------------------|
| `src/ffi/`                | Rust FFI wrapper and error mapping          |
| `src/00_physical/`        | Page allocator, unified memory, GPU kernels |
| `src/01_page/`            | Pages, deltas, TLog, checkpoint/replay      |
| `src/02_semiring/`        | Semiring algebra, delta routing             |
| `src/03_dag/`             | ShadowPageGraph, dependency tracking        |
| `src/04_propagation/`     | CPU/GPU propagation engine                  |
| `src/05_adaptive/`        | Layout optimization, graph rewriting        |
| `src/06_utility/`         | Cost functions, monitoring, telemetry       |
| `../MMSB-top/src/`        | Layers 7-12: Intention → Applications stack |
| `test/`                   | Comprehensive test suite (Layers 0-12)      |
| `examples/`               | Quickstart and tutorial demos               |
| `benchmark/`              | Performance benchmarks and baselines        |
| `docs/`                   | Architecture, API, serialization specs      |
| `project_schedule/`       | Development roadmap and task tracking       |

## Operational Model
- **Semiring discipline** — Deltas as additive merges (`⊕`), propagations as causal applies (`⊗`)
- **Explicit dependencies** — ShadowPageGraph is the only coordination backbone; no hidden globals
- **Command buffers** — All I/O boundaries use command buffers/doorbells for branchless compute
- **Per-state allocators** — Each MMSBState owns its allocator; no global singletons
- **Error propagation** — Rust → RustFFIError → Julia SerializationError chain

## Release Information
- **Current version:** Phase 4 complete (v0.1.0-alpha)
- **Testing:** `julia --project=. -e 'using Pkg; Pkg.test()'` — all passing
- **Examples:** `julia --project=. examples/quickstart.jl`
- **Performance baseline:** See `benchmark/results/baseline.json`
- **Known hotspots:** Propagation ≈500 μs, sparse delta ≈200 μs, alloc ≈5 μs

## Next Phase

**Phase 5: Production Hardening** (7 weeks)
- GPU optimization: Persistent kernels, multi-GPU (NCCL), memory pools
- Performance: SIMD delta merge, lock-free allocation, zero-copy FFI
- Reliability: Error recovery, GPU fallback, memory pressure handling
- Observability: Prometheus metrics, flamegraphs, trace visualization

See `project_schedule/PHASE_5.md` for detailed DAG and task list.

## Documentation

### Core References
- `docs/Architecture.md` — Layer-by-layer system walkthrough
- `docs/API.md` — Public API, configuration, error types
- `docs/SerializationSpec.md` — Binary contract for pages/deltas/checkpoints

### Development
- `project_schedule/DAG_DEPENDENCIES.md` — Phase 1-4 completion status
- `project_schedule/PHASE_5.md` — Production hardening roadmap
- `project_schedule/TASK_LOG_PHASE_5.md` — Detailed task tracking
- `project_schedule/completed/` — Archived phase documentation
- [`../MMSB-top/README.md`](../MMSB-top/README.md) — Intention, reasoning, planning, agents, and application layers

## Requirements

- **Rust:** 1.70+
- **Julia:** 1.12+
- **CUDA:** 12.0+ (optional, for GPU acceleration)
- **OS:** Linux x86_64 (primary target)

## Contributing

1. Run formatting/linting and maintain docstrings
2. Update `docs/*.md` and `examples/*.jl` when changing public API
3. Capture new baselines with `benchmark/benchmarks.jl` for performance changes
4. All tests must pass: `julia --project=. -e 'using Pkg; Pkg.test()'`

## License

APACHE 2.0
