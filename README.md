# MMSB — Memory-Mapped State Bus

Self-optimizing GPU-accelerated memory system covering Layers 0‑6 (physical → utility). MMSB is a deterministic, delta-driven shared-memory fabric that lets CPU, GPU, and compiler subsystems share page-aligned state under a single transaction log. Every operation follows the MMSB semiring law `state × delta → state′`, enabling deterministic replay and algebraic propagation. Reasoning, planning, and agent tooling (Layers 7‑12) now live in the [MMSB-top](../MMSB-top) repository.

## Phase Status
- **Phase 6 Complete** — Core runtime, stress harnesses, and documentation are validated and archived under `project_schedule/completed/`.
- **Phase 7 In Flight** — Production validation week (stress, Julia harness, docs) is tracked in `project_schedule/05_DAG_DEPENDENCIES.md` and `project_schedule/PHASE_7_PLAN.md`.

### Feature Coverage (Phase 6)
| Capability                                      | Status | Notes                                                                                           |
|-------------------------------------------------+--------+-------------------------------------------------------------------------------------------------|
| Lock-free propagation queue + throughput engine | ✅     | Release tests cover single + multi-worker deltas with perf hints (`tests/stress_throughput.rs`) |
| Long-run stability + invariant suite            | ✅     | 10k-cycle perturbation harness exercises DAG + invariant checker (`tests/stress_stability.rs`)  |
| Memory pressure + GC monitor                    | ✅     | 1KB/page budget enforced with GC latency metrics (`tests/stress_memory.rs`)                     |
| Julia validation harness                        | ✅     | `benchmark/run_validation.jl` orchestrates Rust stress tests + `BenchmarkTools` suite           |
| CLI validation summary                          | ✅     | `benchmark/validate_all.jl` emits the ✓/✗ report shown below and propagates exit codes          |
| Documentation refresh                           | ✅     | README tables and Phase DAG updated for the Phase 6→7 hand-off                                  |

### Benchmark Snapshot (release build)
| Benchmark                    | Target                                     | Latest Measurement                                                                                                       |
|------------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------|
| #1 Replay determinism        | divergence `< 1e-9`                        | `benchmark/benchmarks.jl` replay median `≤50ms` (deterministic diff baseline)                                            |
| #5 Throughput                | ≥1M (single) / 10M (multi) deltas/sec      | Single-thread `≈2.2M/sec`, multi-thread `≈2.1M/sec` (10M multi goal tracked in schedule; `perf` guidance in test output) |
| #6 Tick latency              | `<16ms` per tick                           | Full pipeline median `≈13ms` (BenchmarkTools `system/full_pipeline`)                                                     |
| #7 Memory footprint          | avg page ≤1KB, GC `<3ms`, 1M pages `<1GiB` | Avg `1024B`, projected `0.95GiB`, GC `0.017ms`, fragmentation `≈12.5MiB` (see `stress_memory.rs`)                        |
| #8/#9 Stability & invariants | 10k cycles, 0 violations                   | Harness reports `10,000 cycles`, `max divergence ≈0.13`, `0 invariants` (`stress_stability.rs`)                          |
| #10 Provenance               | lookup `<50ms`, depth <32                  | Graph BFS median `≤50ms` (BenchmarkTools `graph/bfs_1024`)                                                               |

Run `julia --project=. benchmark/validate_all.jl` (set `HOME`/`JULIA_DEPOT_PATH` if needed) to execute the combined validation harness, which calls the Rust stress tests and BenchmarkTools suite above.

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
Harness: Validation/CI    → Rust stress_* tests, `benchmark/run_validation.jl`, `benchmark/validate_all.jl`
```

 

## Quick Start

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test --release
HOME=$PWD/.home JULIA_DEPOT_PATH=$PWD/.julia-depot julia --project=. benchmark/validate_all.jl
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
- **Performance baselines:** `benchmark/results/phase6.json` (latest archive) + `benchmark/validate_all.jl` (live stress/benchmark harness)

## Next Steps
- Phase 6 benchmark validation complete (see `benchmark/results/phase6.json` + README snapshot tables).
- Remaining scheduling/architecture docs are archived under `project_schedule/completed/`.
- Use [`MMSB-top`](../MMSB-top) for Layers 7‑12 development.

## Documentation

### Core References
- `docs/Architecture.md` — Layer-by-layer system walkthrough
- `docs/API.md` — Public API, configuration, error types
- `docs/SerializationSpec.md` — Binary contract for pages/deltas/checkpoints

### Development
- `project_schedule/DAG_DEPENDENCIES.md` — Phase 1-4 completion status
- `project_schedule/PHASE_7_PLAN.md` — Current production validation roadmap (Phase 7)
- `project_schedule/TASK_LOG_PHASE_5.md` — Archived Phase 5 task tracking (see `project_schedule/completed/` for Phase 6)
- `project_schedule/completed/` — Archived phase documentation
- [`../MMSB-top/README.md`](../MMSB-top/README.md) — Intention, reasoning, planning, agents, and application layers

## Enhanced File Server

A production-ready HTTP server with advanced query capabilities for serving MMSB project files to AI agents (Grok, Claude, etc).

**Location:** `tools/website/`

### Features

- **Advanced Queries**: Filter by extension, type, search, pattern matching
- **Multiple Formats**: JSON, HTML, text output
- **Sorting**: By name, size, modified date, type
- **Pagination**: Limit/offset and page-based navigation
- **Recursive Traversal**: Depth-limited directory exploration
- **Metadata Queries**: File info without content download
- **Statistics**: Directory aggregation and analysis
- **Performance**: Sub-20ms response times, 175 req/s throughput

### Quick Start

```bash
cd tools/website
node file-server.js
# Server runs on http://127.0.0.1:8889/mmsb
```

### Example Queries

```bash
# Get all Rust files, sorted by modification date
curl "http://127.0.0.1:8889/mmsb/src?ext=.rs&sort=modified&order=desc&format=json"

# Search for test files recursively
curl "http://127.0.0.1:8889/mmsb?pattern=test_*&recursive=true&format=json"

# Get directory statistics
curl "http://127.0.0.1:8889/mmsb/src?stats=true&format=json"

# File metadata only
curl "http://127.0.0.1:8889/mmsb/Cargo.toml?metadata=true&format=json"
```

### Documentation

- **API Reference**: `tools/website/README.md`
- **Performance Analysis**: `tools/website/PERFORMANCE.md`
- **Test Suite**: `tools/website/test-server.sh`
- **Benchmarks**: `tools/website/benchmark-server.sh`

**Production Deployment:** https://cheese-server.duckdns.org/mmsb

## Requirements

- **Rust:** 1.70+
- **Julia:** 1.12+
- **Node.js:** 14.0+ (for file server)
- **CUDA:** 12.0+ (optional, for GPU acceleration)
- **OS:** Linux x86_64 (primary target)

## Contributing

1. Run formatting/linting and maintain docstrings
2. Update `docs/*.md` and `examples/*.jl` when changing public API
3. Capture new baselines with `cargo run --bin phase6_bench --release` (writes to `benchmark/results/phase6.json`) and `benchmark/benchmarks.jl`
4. All tests must pass: `cargo test --lib --tests --no-default-features`

## License

APACHE 2.0
