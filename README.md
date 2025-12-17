# MMSB — Memory-Mapped State Bus

Self-optimizing GPU-accelerated memory system covering Layers 0‑6 (physical → utility). MMSB is a deterministic, delta-driven shared-memory fabric that lets CPU, GPU, and compiler subsystems share page-aligned state under a single transaction log. Every operation follows the MMSB semiring law `state × delta → state′`, enabling deterministic replay and algebraic propagation. Reasoning, planning, and agent tooling (Layers 7‑12) now live in the [MMSB-top](../MMSB-top) repository.



## Why MMSB
- **Deterministic replay** — Byte-level deltas, epochs, and transaction log (TLog) enable checkpoint/reconstruction
- **CPU/GPU coherence** — Unified API across CPU pages, CUDA buffers, and unified memory
- **Declarative graph** — ShadowPageGraph captures dependencies; propagation is algebraic
- **Self-optimization** — Adaptive memory layout, graph rewriting, entropy reduction
- **Reasoning layers externalized** — [`MMSB-top`](../MMSB-top) now hosts Layers 7‑12 (intention, planners, LLM tooling), keeping this repository focused on the core runtime
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

### Basic Usage (Layers 0-6)
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

### Layers 7-12
Reasoning, planning, and agent integrations moved to [`MMSB-top`](../MMSB-top). Use that repository for intention engines, LLM tools, or any logic above the utility layer.

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
| `../MMSB-top/`           | Layers 7-12: Intention → Applications stack (separate repo) |
| `test/`                   | Comprehensive test suite (Layers 0-12)      |
| `examples/`               | Quickstart and tutorial demos               |
| `benchmark/`              | Performance benchmarks and baselines        |
| `docs/`                   | Architecture, API, serialization specs      |
| `project_schedule/`       | Development roadmap and task tracking       |

### Layer Benchmarks
Each layer now has a co-located benchmark guide under `benchmark/<layer>/README.md`:
- `benchmark/00_physical/` — allocation, unified memory, device sync hot paths
- `benchmark/01_page/` — delta pipelines, TLog replay, checkpointing
- `benchmark/02_semiring/` — semiring cores and purity validation
- `benchmark/03_dag/` — graph validators, traversal workloads
- `benchmark/04_propagation/` — ring buffer, throughput engine, tick latency
- `benchmark/05_adaptive/` — layout optimizer and clustering heuristics
- `benchmark/06_utility/` — telemetry, invariant, memory monitor suites

## Operational Model
- **Semiring discipline** — Deltas as additive merges (`⊕`), propagations as causal applies (`⊗`)
- **Explicit dependencies** — ShadowPageGraph is the only coordination backbone; no hidden globals
- **Command buffers** — All I/O boundaries use command buffers/doorbells for branchless compute
- **Per-state allocators** — Each MMSBState owns its allocator; no global singletons
- **Error propagation** — Rust → RustFFIError → Julia SerializationError chain

## Release Information
- **Current version:** Phase 6 validation (v0.2.0-alpha)
- **Testing:** `cargo test --lib --tests --no-default-features`
- **Examples:** `julia --project=. examples/quickstart.jl`
- **Performance baselines:** `benchmark/results/baseline.json` (legacy) and `benchmark/results/phase6.json` (latest Phase 6 run)

## Next Steps
- Phase 6 benchmark validation complete (see `benchmark/results/phase6.json`).
- Remaining scheduling/architecture docs are archived under `project_schedule/completed/`.
- Use [`MMSB-top`](../MMSB-top) for Layers 7‑12 development.

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
3. Capture new baselines with `cargo run --bin phase6_bench --release` (writes to `benchmark/results/phase6.json`) and `benchmark/benchmarks.jl`
4. All tests must pass: `cargo test --lib --tests --no-default-features`

## License

APACHE 2.0
