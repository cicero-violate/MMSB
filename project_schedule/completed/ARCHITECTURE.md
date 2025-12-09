# MMSB Architecture Specification v2.0

## Full 13-Layer Architecture

```
LAYER 0  — Physical Memory Layer       → Rust
LAYER 1  — Page Layer                  → Rust
LAYER 2  — Semiring Algebra            → Rust core, Julia config
LAYER 3  — ShadowGraph (DAG Logic)     → Rust core, Julia DSL
LAYER 4  — Propagation Engine          → Rust + CUDA
LAYER 5  — Adaptive Memory Layout      → Rust exec, Julia logic
LAYER 6  — Utility Engine              → Julia
LAYER 7  — Intention Engine            → Julia
LAYER 8  — Reasoning Engine            → Julia (NEW)
LAYER 9  — Planning Engine             → Julia (NEW)
LAYER 10 — Agent Interface             → Julia
LAYER 11 — External Agents             → Julia/Python/RL
LAYER 12 — Applications                → Any language
```

---

## Latent Mathematical Structure

Let:
- $\mathcal{S}$ = State space (pages + checkpoints)
- $\mathcal{P} = \{p_i\}$ = Set of pages
- $\mathcal{D} = \{\Delta_i\}$ = Set of deltas
- $\mathcal{G} = (V, E)$ = ShadowGraph (DAG)
- $\mathbb{K} = (\oplus, \otimes, 0, 1)$ = Semiring structure
- $C: \mathcal{S} \to \mathbb{R}$ = Cost function
- $U = -C$ = Utility function
- $I = \arg\max_g U$ = Intention signal
- $\Phi: \mathcal{S} \to \mathcal{R}$ = Reasoning function
- $\Pi: \mathcal{R} \times I \to \mathcal{A}^*$ = Planning function

**Core Equations:**

State update: $S_{t+1} = S_t \oplus \Delta_t$

DAG propagation: $\Delta_v = \bigoplus_{u \in \text{parents}(v)} (\Delta_u \otimes W_{u \to v})$

Utility: $U(S, M, \mathcal{G}) = -\sum_i w_i \cdot \text{cost}_i(S, M, \mathcal{G})$

Reasoning: $\Phi(S) = \text{Infer}(\mathcal{G}, \mathcal{P}, \text{Rules})$

Planning: $\Pi(\Phi(S), I) = \text{Search}(\text{goal}=I, \text{state}=\Phi(S))$

---

## Layer-by-Layer File Organization

### LAYER 0: Physical Memory (Hardware Interface)

**Folder:** `src/00_physical/`

| File                  | Language | Purpose                                         |
|-----------------------+----------+-------------------------------------------------|
| `allocator.rs`        | Rust     | Core page allocator with CPU/GPU/Unified memory |
| `allocator_stats.rs`  | Rust     | Allocation telemetry and statistics             |
| `device.rs`           | Rust     | GPU device management and enumeration           |
| `device_registry.rs`  | Rust     | Device buffer registry                          |
| `host_device_sync.rs` | Rust     | CPU ↔ GPU synchronization primitives            |
| `mod.rs`              | Rust     | Module exports                                  |
| `PageAllocator.jl`    | Julia    | High-level allocator API                        |
| `UnifiedMemory.jl`    | Julia    | CUDA unified memory wrapper                     |
| `DeviceSync.jl`       | Julia    | Device sync orchestration                       |
| `GPUKernels.jl`       | Julia    | Zero-copy GPU kernel wrappers                   |

**Key Responsibilities:**
- Page allocation/deallocation
- GPU unified memory management
- Zero-copy buffer creation
- Device synchronization
- Memory residency tracking

---

### LAYER 1: Page Layer (State Representation)

**Folder:** `src/01_page/`

| File                    | Language | Purpose                                  |
|-------------------------+----------+------------------------------------------|
| `page.rs`               | Rust     | Core Page structure with Arc/RwLock      |
| `delta.rs`              | Rust     | Delta structure and merge logic          |
| `epoch.rs`              | Rust     | Epoch versioning (AtomicU32)             |
| `checkpoint.rs`         | Rust     | Checkpoint serialization/deserialization |
| `simd_mask.rs`          | Rust     | SIMD mask generation for deltas          |
| `delta_merge.rs`        | Rust     | Delta compression and merging            |
| `tlog.rs`               | Rust     | Transaction log core                     |
| `tlog_compression.rs`   | Rust     | Delta compaction algorithms              |
| `tlog_serialization.rs` | Rust     | TLog binary serialization                |
| `tlog_replay.rs`        | Rust     | TLog replay engine                       |
| `mod.rs`                | Rust     | Module exports                           |
| `Page.jl`               | Julia    | Page API and utilities                   |
| `Delta.jl`              | Julia    | Delta creation and inspection            |
| `TLog.jl`               | Julia    | Transaction log interface                |
| `ReplayEngine.jl`       | Julia    | Replay orchestration and verification    |

**Key Responsibilities:**
- Page lifecycle management
- Delta application and merging
- Checkpointing state snapshots
- Transaction log persistence
- Replay for time-travel debugging

---

### LAYER 2: Semiring Algebra (Core Math)

**Folder:** `src/02_semiring/`

| File                    | Language | Purpose                                                      |
|-------------------------+----------+--------------------------------------------------------------|
| `semiring_types.rs`     | Rust     | Semiring trait definitions                                   |
| `semiring_ops.rs`       | Rust     | Core $\oplus$ and $\otimes$ operations                       |
| `standard_semirings.rs` | Rust     | Pre-built semirings (tropical, boolean, etc)                 |
| `mod.rs`                | Rust     | Module exports                                               |
| `Semiring.jl`           | Julia    | Semiring algebra DSL                                         |
| `SemiringConfig.jl`     | Julia    | Custom semiring configuration                                |
| `DeltaRouter.jl`        | Julia    | **MOVED FROM RUNTIME** - Uses semiring $\otimes$ for routing |

**Key Responsibilities:**
- Define state combination ($\oplus$)
- Define delta application/routing ($\otimes$)
- Zero and identity elements
- Algebraic validation
- Custom semiring creation

**Notes:**
- DeltaRouter uses semiring operations for propagation
- Semiring determines how deltas compose and propagate

---

### LAYER 3: ShadowGraph / DAG (Declarative Logic)

**Folder:** `src/03_dag/`

| File                        | Language | Purpose                                      |
|-----------------------------+----------+----------------------------------------------|
| `shadow_graph.rs`           | Rust     | Core DAG structure (adjacency lists)         |
| `shadow_graph_mod.rs`       | Rust     | Module organization                          |
| `shadow_graph_traversal.rs` | Rust     | Topological sort, DFS, BFS                   |
| `edge_types.rs`             | Rust     | Edge type definitions                        |
| `cycle_detection.rs`        | Rust     | DAG validation                               |
| `mod.rs`                    | Rust     | Module exports                               |
| `ShadowPageGraph.jl`        | Julia    | High-level graph API                         |
| `DependencyGraph.jl`        | Julia    | Dependency management                        |
| `EventSystem.jl`            | Julia    | Event emission for graph changes             |
| `GraphDSL.jl`               | Julia    | **NEW** - Declarative graph construction DSL |

**Key Responsibilities:**
- DAG structure maintenance
- Dependency tracking
- Topological ordering for propagation
- Cycle detection
- Edge weight management
- Event-driven updates

---

### LAYER 4: Propagation Engine (Graph Evaluation)

**Folder:** `src/04_propagation/`

| File                            | Language | Purpose                           |
|---------------------------------+----------+-----------------------------------|
| `propagation_engine.rs`         | Rust     | Core propagation logic            |
| `propagation_fastpath.rs`       | Rust     | Fast-path optimization            |
| `propagation_queue.rs`          | Rust     | Queue-based scheduling            |
| `propagation_command_buffer.rs` | Rust     | Command buffering                 |
| `sparse_message_passing.rs`     | Rust     | **NEW** - Sparse propagation      |
| `gpu_propagation.cu`            | CUDA     | **NEW** - GPU propagation kernels |
| `mod.rs`                        | Rust     | Module exports                    |
| `PropagationEngine.jl`          | Julia    | High-level propagation API        |
| `PropagationScheduler.jl`       | Julia    | **NEW** - Intelligent scheduling  |

**Key Responsibilities:**
- Apply semiring algebra over DAG
- Sparse message passing
- GPU-accelerated propagation
- Topological execution ordering
- Fast-path detection
- Queue management

---

### LAYER 5: Adaptive Memory Layout (Self-Optimization)

**Folder:** `src/05_adaptive/`

| File                    | Language | Purpose                              |
|-------------------------+----------+--------------------------------------|
| `memory_layout.rs`      | Rust     | **NEW** - Memory layout optimization |
| `page_clustering.rs`    | Rust     | **NEW** - Hot page clustering        |
| `locality_optimizer.rs` | Rust     | **NEW** - Cache locality improvement |
| `mod.rs`                | Rust     | Module exports                       |
| `AdaptiveLayout.jl`     | Julia    | **NEW** - Adaptive orchestration     |
| `GraphRewriting.jl`     | Julia    | **NEW** - DAG edge rewriting         |
| `EntropyReduction.jl`   | Julia    | **NEW** - Entropy minimization       |
| `LocalityAnalysis.jl`   | Julia    | **NEW** - Access pattern analysis    |

**Key Responsibilities:**
- Reorder pages for locality
- Rewrite graph edges for efficiency
- Cluster frequently accessed pages
- Reduce replay cost
- Minimize entropy
- Improve cache utilization

---

### LAYER 6: Utility Engine (Evaluation Function)

**Folder:** `src/06_utility/`

| File                 | Language | Purpose                                    |
|----------------------+----------+--------------------------------------------|
| `cost_functions.jl`  | Julia    | **NEW** - Cost function definitions        |
| `utility_engine.jl`  | Julia    | **NEW** - $U = -C$ computation             |
| `telemetry.rs`       | Rust     | **NEW** - Performance telemetry            |
| `entropy_measure.jl` | Julia    | **NEW** - Entropy/density metrics          |
| `Monitoring.jl`      | Julia    | System monitoring (moved from utils)       |
| `CostAggregation.jl` | Julia    | **NEW** - Multi-objective cost aggregation |

**Key Responsibilities:**
- Measure propagation cost
- Track memory fragmentation
- Compute graph complexity
- Calculate replay efficiency
- Aggregate utility scores
- Drive adaptation decisions

**Cost Functions:**
- Locality cost: $C_{\text{loc}} = \sum_i d(p_i, p_j)$ (cache misses)
- Propagation cost: $C_{\text{prop}} = \sum_{e \in E} w_e \cdot \text{freq}(e)$
- Delta density: $C_{\text{delta}} = \frac{|\text{deltas}|}{|\text{pages}|}$
- Graph load: $C_{\text{graph}} = |E| + \text{depth}(\mathcal{G})$

---

### LAYER 7: Intention Engine (Goal Emergence)

**Folder:** `src/07_intention/`

| File                        | Language | Purpose                                   |
|-----------------------------+----------+-------------------------------------------|
| `intention_engine.jl`       | Julia    | **NEW** - $I = \arg\max_g U$              |
| `goal_emergence.jl`         | Julia    | **NEW** - Goal formation                  |
| `structural_preferences.jl` | Julia    | **NEW** - System preferences              |
| `attractor_states.jl`       | Julia    | **NEW** - Stable attractor identification |
| `IntentionTypes.jl`         | Julia    | **NEW** - Intention type system           |

**Key Responsibilities:**
- Compute $I = \arg\max_g U(S, M, g)$
- Generate internal goals
- Form structural preferences
- Identify stable attractors
- Drive routing improvements
- Evolve memory layout

**Not neural. Not backprop. System-level intention from utility maximization.**

---

### LAYER 8: Reasoning Engine (Symbolic + Structural) **NEW**

**Folder:** `src/08_reasoning/`

| File                        | Language | Purpose                                |
|-----------------------------+----------+----------------------------------------|
| `reasoning_engine.jl`       | Julia    | **NEW** - Main reasoning orchestration |
| `structural_inference.jl`   | Julia    | **NEW** - Infer structure from DAG     |
| `constraint_propagation.jl` | Julia    | **NEW** - Constraint solver            |
| `dependency_inference.jl`   | Julia    | **NEW** - Causal reconstruction        |
| `pattern_formation.jl`      | Julia    | **NEW** - Symbolic pattern matching    |
| `rule_evaluation.jl`        | Julia    | **NEW** - Rule-based reasoning         |
| `logic_engine.jl`           | Julia    | **NEW** - First-order logic            |
| `ReasoningTypes.jl`         | Julia    | **NEW** - Type system for reasoning    |

**Key Responsibilities:**
- Structural inference from DAG
- Logical reasoning over pages
- Constraint propagation
- Dependency inference
- Causal reconstruction
- Symbolic pattern formation
- Rule evaluation

**Reasoning consumes:**
- MMSB pages (working memory)
- DAG structure (causal graph)
- Utility (optimization signal)
- Intention (goal context)

**Reasoning produces:**
- Inferred dependencies
- Symbolic constraints
- Causal models
- Pattern hypotheses

**Think:** Graph reasoner + Constraint solver + Symbolic engine

---

### LAYER 9: Planning Engine (Search + Optimization) **NEW**

**Folder:** `src/09_planning/`

| File                       | Language | Purpose                                    |
|----------------------------+----------+--------------------------------------------|
| `planning_engine.jl`       | Julia    | **NEW** - Main planning orchestration      |
| `search_algorithms.jl`     | Julia    | **NEW** - A*, MCTS, beam search            |
| `goal_decomposition.jl`    | Julia    | **NEW** - Hierarchical goal breakdown      |
| `strategy_generation.jl`   | Julia    | **NEW** - Multi-step strategy synthesis    |
| `rollout_simulation.jl`    | Julia    | **NEW** - Forward simulation               |
| `decision_graphs.jl`       | Julia    | **NEW** - Decision tree/graph construction |
| `rl_planning.jl`           | Julia    | **NEW** - RL-based planning (Enzyme/Lux)   |
| `optimization_planning.jl` | Julia    | **NEW** - Optimization-based planning      |
| `PlanningTypes.jl`         | Julia    | **NEW** - Type system for plans            |

**Key Responsibilities:**
- Search (A*, MCTS, beam)
- Optimization (gradient-based via Enzyme)
- Rollout simulation
- Decision trees
- Multi-step foresight
- RL-like planning
- Goal decomposition
- Strategy generation

**Planning consumes:**
- Reasoning output (symbolic constraints, causal models)
- Intention (goals to achieve)
- Utility (evaluation function)

**Planning produces:**
- Action sequences
- Deltas for MMSB
- New intentions
- Subgoals

**Where Enzyme, Lux, symbolic planning, MCTS live.**

---

### LAYER 10: Agent Interface (Brain API)

**Folder:** `src/10_agent_interface/`

| File                        | Language | Purpose                                |
|-----------------------------+----------+----------------------------------------|
| `MMSBState.jl`              | Julia    | Agent-visible state (moved from types) |
| `API.jl`                    | Julia    | Main agent API (moved from root)       |
| `BaseHook.jl`               | Julia    | Hook system for agents                 |
| `CoreHooks.jl`              | Julia    | Core hook implementations              |
| `CompilerHooks.jl`          | Julia    | Compiler integration                   |
| `InstrumentationManager.jl` | Julia    | Instrumentation orchestration          |
| `checkpoint_api.jl`         | Julia    | **NEW** - Checkpoint API for agents    |
| `event_subscription.jl`     | Julia    | **NEW** - Event subscription           |
| `AgentProtocol.jl`          | Julia    | **NEW** - Standard agent protocol      |

**Key Responsibilities:**
- Read pages
- Write deltas
- Subscribe to events
- Checkpoint states
- Create external goals
- Apply Enzyme/RL
- Instrumentation hooks

**Agents can:**
- Query MMSB state
- Mutate via deltas
- Listen to events
- Checkpoint/restore
- Set goals

---

### LAYER 11: External Agents (Learning Systems)

**Folder:** `src/11_agents/`

| File                    | Language | Purpose                             |
|-------------------------+----------+-------------------------------------|
| `rl_agent.jl`           | Julia    | **NEW** - RL-based agents           |
| `symbolic_agent.jl`     | Julia    | **NEW** - Symbolic reasoning agents |
| `enzyme_integration.jl` | Julia    | **NEW** - Enzyme.jl differentiation |
| `lux_models.jl`         | Julia    | **NEW** - Lux.jl neural models      |
| `planning_agent.jl`     | Julia    | **NEW** - Planning/search agents    |
| `hybrid_agent.jl`       | Julia    | **NEW** - Hybrid symbolic+neural    |
| `AgentTypes.jl`         | Julia    | **NEW** - Agent type system         |

**Key Responsibilities:**
- RL agents using MMSB as world model
- Symbolic reasoning agents
- Enzyme.jl for differentiation
- Lux/Zygote for neural components
- Planning agents using Layer 9

**Agents learn. MMSB hosts their world model.**

---

### LAYER 12: Applications (User-Facing Systems)

**Folder:** `src/12_applications/`

| File                         | Language | Purpose                                |
|------------------------------+----------+----------------------------------------|
| `llm_tools.jl`               | Julia    | **NEW** - LLM integration tools        |
| `world_simulation.jl`        | Julia    | **NEW** - World model applications     |
| `multi_agent_system.jl`      | Julia    | **NEW** - Multi-agent orchestration    |
| `financial_modeling.jl`      | Julia    | **NEW** - Financial applications       |
| `memory_driven_reasoning.jl` | Julia    | **NEW** - Memory-driven reasoning apps |
| `examples/`                  | Julia    | **NEW** - Example applications         |

---

## FFI / Support Infrastructure

**Folder:** `src/ffi/`

| File            | Language | Purpose                        |
|-----------------+----------+--------------------------------|
| `ffi.rs`        | Rust     | FFI boundary (C ABI)           |
| `mod.rs`        | Rust     | Module exports                 |
| `FFIWrapper.jl` | Julia    | Julia FFI wrapper              |
| `RustErrors.jl` | Julia    | Error marshaling               |
| `Errors.jl`     | Julia    | Error types (moved from types) |

**Root Files:**
- `lib.rs` - Rust root library
- `MMSB.jl` - Julia root module
- `test.jl` - Test suite

---

## Cross-Layer Dependencies

```
Layer 12 (Apps)
    ↓
Layer 11 (External Agents)
    ↓
Layer 10 (Agent Interface)
    ↓
Layer 9 (Planning) ← consumes reasoning + intention
    ↓
Layer 8 (Reasoning) ← consumes DAG + pages + utility
    ↓
Layer 7 (Intention) ← consumes utility
    ↓
Layer 6 (Utility) ← measures everything below
    ↓
Layer 5 (Adaptive) ← uses utility to optimize
    ↓
Layer 4 (Propagation) ← executes DAG + semiring
    ↓
Layer 3 (DAG/ShadowGraph) ← defines program
    ↓
Layer 2 (Semiring) ← defines algebra
    ↓
Layer 1 (Page Layer) ← state representation
    ↓
Layer 0 (Physical Memory) ← hardware interface
```

**Key Insight:** Lower layers are performance-critical (Rust). Upper layers are cognitive (Julia). Middle layers are hybrid.

---

## Language Distribution Summary

| Layer                |   Rust |  Julia |  CUDA |   Total |
|----------------------+--------+--------+-------+---------|
| 0 - Physical         |      6 |      4 |     0 |      10 |
| 1 - Page             |     11 |      4 |     0 |      15 |
| 2 - Semiring         |      4 |      3 |     0 |       7 |
| 3 - DAG              |      6 |      4 |     0 |      10 |
| 4 - Propagation      |      6 |      2 |     1 |       9 |
| 5 - Adaptive         |      4 |      4 |     0 |       8 |
| 6 - Utility          |      1 |      6 |     0 |       7 |
| 7 - Intention        |      0 |      5 |     0 |       5 |
| 8 - Reasoning        |      0 |      8 |     0 |       8 |
| 9 - Planning         |      0 |      9 |     0 |       9 |
| 10 - Agent Interface |      0 |      9 |     0 |       9 |
| 11 - External Agents |      0 |      7 |     0 |       7 |
| 12 - Applications    |      0 |      6 |     0 |       6 |
| FFI/Support          |      3 |      3 |     0 |       6 |
| **TOTAL**            | **41** | **74** | **1** | **116** |

**Rust:** 41 files (35.3%) - Performance-critical infrastructure
**Julia:** 74 files (63.8%) - High-level orchestration and cognition
**CUDA:** 1 file (0.9%) - GPU propagation kernels

---

## Implementation Priority

1. **Phase 1:** Layers 0-4 (Core infrastructure) - 6 weeks
2. **Phase 2:** Layers 5-7 (Self-optimization) - 4 weeks
3. **Phase 3:** Layers 8-9 (Cognition) - 6 weeks
4. **Phase 4:** Layers 10-12 (Agents + Apps) - 4 weeks

**Total:** 20 weeks to complete architecture

---

**MMSB is a GPU-accelerated, semiring-based, delta-driven DAG state machine with adaptive memory, utility-based intention, symbolic reasoning, planning, and agent integration—forming a complete non-neural AGI substrate.**
