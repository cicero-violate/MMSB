# MMSB - Memory-Mapped State Bus

Self-optimizing GPU-accelerated memory system with autonomous reasoning and planning capabilities.

## Status

**Phase 4 Complete** ✓ (2025-12-09)
- All 13 layers operational
- Full test suite passing
- Production ready

## Architecture

```
Layer 12: Applications     → LLM, WorldSim, MultiAgent, Finance
Layer 11: External Agents  → RL, Symbolic, Planning, Hybrid
Layer 10: Agent Interface  → Checkpoint, Events, Protocol
Layer 9:  Planning         → MCTS, GoalDecomp, RL, Enzyme
Layer 8:  Reasoning        → Inference, Constraints, Logic
Layer 7:  Intention        → Goals, Preferences, Attractors  
Layer 6:  Utility          → Costs, Telemetry, Entropy
Layer 5:  Adaptive Memory  → Layout, Clustering, Rewriting
Layer 4:  Propagation      → CPU/GPU message passing
Layer 3:  DAG/Graph        → Dependency tracking, cycles
Layer 2:  Semiring Algebra → Delta routing, merge ops
Layer 1:  Page Layer       → Pages, Deltas, TLog, Checkpoint
Layer 0:  Physical Memory  → Allocator, UnifiedMem, GPU kernels
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

### Usage
```julia
using MMSB

# Create state
state = mmsb_start(enable_gpu=true)

# Allocate pages
page = create_page(state, size=1024, location=:gpu)

# Update with delta
update_page(state, page.id, rand(UInt8, 1024))

# Read back
data = query_page(state, page.id)

# Checkpoint
mmsb_stop(state, checkpoint_path="state.ckpt")
```

## Project Structure

```
src/
├── ffi/           # Rust FFI wrapper
├── 00_physical/   # Allocator, GPU kernels
├── 01_page/       # Pages, deltas, TLog
├── 02_semiring/   # Algebra operations
├── 03_dag/        # Dependency graphs
├── 04_propagation/# Message passing
├── 05_adaptive/   # Self-optimization
├── 06_utility/    # Cost functions
├── 07_intention/  # Goal emergence
├── 08_reasoning/  # Inference engine
├── 09_planning/   # MCTS, strategies
├── 10_agent_interface/ # API, events
├── 11_agents/     # RL, symbolic agents
└── 12_applications/ # Domain apps

test/              # Comprehensive test suite
project_schedule/  # Development roadmap
```

## Next Phase

**Phase 5: Production Hardening** (7 weeks)
- GPU optimization (persistent kernels, multi-GPU)
- Performance (SIMD, lock-free, zero-copy)
- Reliability (error recovery, failover)
- Observability (metrics, profiling)

See `project_schedule/PHASE_5.md` for details.

## Documentation

- `project_schedule/DAG_DEPENDENCIES.md` - Current status
- `project_schedule/PHASE_5.md` - Next phase roadmap
- `project_schedule/completed/` - Archived phase docs

## Requirements

- Rust 1.70+
- Julia 1.12+
- CUDA 12.0+ (optional, for GPU)
- Linux x86_64

## License

MIT
