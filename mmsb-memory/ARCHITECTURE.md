## Proposed mmsb-memory Architecture

### Variables

$$
\begin{align}
\text{SourceTree} &: \text{mmsb-core/src} \\
\text{TargetTree} &: \text{mmsb-memory/src} \\
\text{Pipelines} &: \{\text{Structural}, \text{State}\} \\
\text{Proofs} &: \{D, E, F\} \\
\end{align}
$$

### Latent Equations

$$
\begin{align}
\text{AdmissionGate} &\xrightarrow{\text{verify}(C)} \text{AdmissionProof}(D) \\
\text{CommitEngine} &\xrightarrow{\text{seal}} \text{CommitProof}(E) \\
\text{OutcomeTracker} &\xrightarrow{\text{observe}} \text{OutcomeProof}(F) \\
\end{align}
$$

### Proposed Directory Structure

```
mmsb-memory/src/
├── memory_engine.rs                    # MemoryModule (implements AdmissionStage, CommitStage, OutcomeStage)
├── lib.rs                       # Public API surface
│
├── admission/                   # Admission Gateway (produces D)
│   ├── mod.rs
│   ├── admission_gate.rs       # Verifies JudgmentProof (C), checks epoch validity
│   └── replay_protection.rs    # Prevents duplicate commits
│
├── truth/                       # Truth Ownership & Invariants
│   ├── mod.rs
│   ├── invariant_checker.rs    # From 06_utility/
│   ├── provenance_tracker.rs   # From 06_utility/
│   └── canonical_time.rs       # Epoch and logical time management
│
├── delta/                       # Delta Management (State Pipeline)
│   ├── mod.rs
│   ├── delta_types.rs          # From 01_types/
│   ├── delta.rs                # From 01_page/
│   ├── delta_validation.rs     # From 01_page/
│   ├── delta_merge.rs          # From 01_page/
│   └── columnar_delta.rs       # From 01_page/
│
├── structural/                  # Structural Changes (Structural Pipeline)
│   ├── mod.rs
│   ├── structural_types.rs     # From 03_dag/
│   ├── shadow_graph.rs         # From 03_dag/
│   ├── shadow_graph_mod.rs     # From 03_dag/
│   ├── shadow_graph_traversal.rs # From 03_dag/
│   ├── cycle_detection.rs      # From 03_dag/
│   └── graph_validator.rs      # From 03_dag/
│
├── dag/                         # Dependency Graph (Truth State)
│   ├── mod.rs
│   ├── dependency_graph.rs     # From 03_dag/
│   ├── graph_trait.rs          # From 03_dag/
│   ├── edge_types.rs           # From 03_dag/
│   └── dag_snapshot.rs         # From 03_dag/
│
├── commit/                      # Commit Engine (produces E)
│   ├── mod.rs
│   ├── commit_engine.rs        # New - orchestrates commit
│   ├── page_commit.rs          # From 01_page/
│   ├── dag_commit.rs           # From 03_dag/
│   └── commit_validator.rs     # Structural invariants validation
│
├── page/                        # Page Management
│   ├── mod.rs
│   ├── page_types.rs           # From 01_types/
│   ├── page.rs                 # From 01_page/
│   ├── allocator.rs            # From 01_page/
│   ├── lockfree_allocator.rs   # From 01_page/
│   └── device_registry.rs      # From 01_page/
│
├── epoch/                       # Epoch Management
│   ├── mod.rs
│   ├── epoch_types.rs          # From 01_types/
│   ├── epoch.rs                # From 01_page/
│   ├── checkpoint.rs           # From 01_page/
│   └── gc.rs                   # From 01_types/
│
├── tlog/                        # Transaction Log (State Pipeline)
│   ├── mod.rs
│   ├── tlog.rs                 # From 01_page/
│   ├── tlog_serialization.rs   # From 01_page/
│   ├── tlog_compression.rs     # From 01_page/
│   └── tlog_replay.rs          # From 01_page/
│
├── replay/                      # Deterministic Replay
│   ├── mod.rs
│   ├── replay_engine.rs        # New - orchestrates replay
│   ├── replay_validator.rs     # From 01_page/
│   └── dag_log.rs              # From 03_dag/
│
├── outcome/                     # Outcome Tracking (produces F)
│   ├── mod.rs
│   ├── outcome_tracker.rs      # New - tracks execution results
│   ├── dag_validator.rs        # From 03_dag/ - post-commit validation
│   └── dag_errors.rs           # From 03_dag/
│
├── propagation/                 # State Propagation (post-commit)
│   ├── mod.rs
│   ├── propagation_engine.rs   # From 04_propagation/
│   ├── propagation_fastpath.rs # From 04_propagation/
│   ├── propagation_queue.rs    # From 04_propagation/
│   ├── dag_propagation.rs      # From 04_propagation/
│   ├── sparse_message_passing.rs # From 04_propagation/
│   ├── ring_buffer.rs          # From 04_propagation/
│   ├── throughput_engine.rs    # From 04_propagation/
│   ├── tick_orchestrator.rs    # From 04_propagation/
│   └── propagation_command_buffer.rs # From 04_propagation/
│
├── materialization/             # Pure Materialization
│   ├── mod.rs
│   └── pure_materialize.rs     # From 03_materialization/
│
├── semiring/                    # Semiring Operations
│   ├── mod.rs
│   ├── semiring_types.rs       # From 02_semiring/
│   ├── semiring_ops.rs         # From 02_semiring/
│   ├── standard_semirings.rs   # From 02_semiring/
│   └── purity_validator.rs     # From 02_semiring/
│
├── physical/                    # Physical Memory Management
│   ├── mod.rs
│   ├── allocator_stats.rs      # From 00_physical/
│   ├── gpu_memory_pool.rs      # From 00_physical/
│   └── nccl_integration.rs     # From 00_physical/
│
├── optimization/                # Memory Layout Optimization
│   ├── mod.rs
│   ├── memory_layout.rs        # From 06_optimization/
│   ├── locality_optimizer.rs   # From 06_optimization/
│   └── page_clustering.rs      # From 06_optimization/
│
├── device/                      # Device Management
│   ├── mod.rs
│   ├── device.rs               # From 01_page/
│   ├── host_device_sync.rs     # From 01_page/
│   ├── simd_mask.rs            # From 01_page/
│   └── integrity_checker.rs    # From 01_page/
│
└── proofs/                      # Proof Production (D, E, F)
    ├── mod.rs
    ├── admission_proof.rs      # Produces AdmissionProof (D)
    ├── commit_proof.rs         # Produces CommitProof (E)
    └── outcome_proof.rs        # Produces OutcomeProof (F)
```

### Files NOT Moved to mmsb-memory

**To mmsb-learning:**
- `07_adaptive/*` (all files)

**To mmsb-service:**
- `06_utility/telemetry.rs`
- `06_utility/memory_monitor.rs`
- `06_utility/cpu_features.rs`

**To mmsb-proof (structure definitions only):**
- `05_proof/structural_proof.rs` (schema only)
- `05_proof/execution_proof.rs` (schema only)
- `05_proof/admission_proof.rs` (schema only - may already exist)

**To mmsb-authenticate (verification logic):**
- `05_proof/delta_stream.rs` (proof chaining logic)

**Not Moved (integration/tools):**
- `bin/*`
- `ffi/*`
- `ffi.rs`
- `julia/*`
- Top-level markdown docs (keep in mmsb-core for reference)

### Key Semantic Domains

1. **Admission** - Gate that verifies JudgmentProof and admits execution
2. **Truth** - Invariants, provenance, canonical time ownership
3. **Delta & Structural** - Two parallel pipelines for state vs structure changes
4. **DAG** - Dependency graph as canonical truth
5. **Commit** - Sealing mutations with CommitProof
6. **Outcome** - Tracking execution results with OutcomeProof
7. **Replay** - Deterministic replay from persisted proofs
8. **Propagation** - Post-commit state propagation
9. **Physical/Device** - Low-level memory and device management

### Module Implementation

The `memory-engine.rs` will implement:
- `AdmissionStage` → produces `AdmissionProof (D)`
- `CommitStage` → produces `CommitProof (E)`
- `OutcomeStage` → produces `OutcomeProof (F)`
- `EventHandler<ExecutionRequested>`


   
