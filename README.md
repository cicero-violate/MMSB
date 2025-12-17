# MMSB — Memory-Mapped State Bus

Self-optimizing GPU-accelerated memory system with autonomous reasoning and planning. MMSB is a deterministic, delta-driven shared-memory fabric that lets CPU, GPU, and compiler subsystems share page-aligned state under a single transaction log. Every operation follows the MMSB semiring law `state × delta → state′`, enabling deterministic replay and algebraic propagation.

## Status

**Phase 4 Complete** ✓ (2025-12-09)
- All 13 layers operational
- Full test suite passing
- Production ready

## Why MMSB
- **Deterministic replay** — Byte-level deltas, epochs, and transaction log (TLog) enable checkpoint/reconstruction
- **CPU/GPU coherence** — Unified API across CPU pages, CUDA buffers, and unified memory
- **Declarative graph** — ShadowPageGraph captures dependencies; propagation is algebraic
- **Self-optimization** — Adaptive memory layout, graph rewriting, entropy reduction
- **Autonomous reasoning** — Structural inference, constraint propagation, goal emergence
- **Planning & agents** — MCTS, RL integration, symbolic/hybrid agents
- **Instrumentation** — Julia compiler hooks for SSA/IR provenance
- **Observability** — Built-in monitoring: allocator pressure, delta latency, propagation metrics

## Architecture (6 Layers)

```
Layer 12: Applications     → LLM Tools, World Simulation, Multi-Agent, Finance
Layer 11: External Agents  → RL Agent, Symbolic Agent, Planning Agent, Hybrid
Layer 10: Agent Interface  → Checkpoint API, Event Subscription, Agent Protocol
Layer 9:  Planning Engine  → MCTS, Goal Decomposition, RL Planning, Enzyme Integration
Layer 8:  Reasoning Engine → Structural Inference, Constraints, Logic, Patterns
Layer 7:  Intention Engine → Goal Emergence, Preferences, Attractor States
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

### Advanced: Agents & Planning
```julia
using MMSB

state = mmsb_start()

# Create hybrid agent (symbolic + RL)
agent = HybridAgent(0.7)  # 70% RL, 30% symbolic

# Observe state
obs = observe(agent, state)

# Plan actions
actions = plan(agent, obs)

# World simulation
world = World(state, 0.01)  # dt=10ms
entity = add_entity!(world, :robot, Dict(:x => 0.0, :y => 0.0))
simulate_step!(world)
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
| `src/07_intention/`       | Goal emergence, attractor states            |
| `src/08_reasoning/`       | Inference engine, constraints, logic        |
| `src/09_planning/`        | MCTS, RL planning, Enzyme integration       |
| `src/10_agent_interface/` | Checkpoint API, events, protocol            |
| `src/11_agents/`          | RL/symbolic/planning/hybrid agents          |
| `src/12_applications/`    | LLM tools, world sim, multi-agent, finance  |
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
