# MMSB Canonical Bus Architecture

## Bus Definitions

### JudgmentBus
**Purpose:** Decision flow - authority approval chain
**Proof Chain:** A → B → C
**Modules:**
```
mmsb-intent (produces A: IntentProof)
└── mmsb-policy (produces B: PolicyProof)
    └── mmsb-judgment (produces C: JudgmentProof - SOLE AUTHORITY)
```

### ExecutionBus
**Purpose:** Mechanical execution flow - approved operations
**Flow:** ExecutionRequest → Execution → Outcome
**Modules:**
```
mmsb-executor (requests execution)
└── mmsb-device (physical execution)
    └── mmsb-propagation (state propagation)
```

### StateBus
**Purpose:** Sealed canonical truth - admission and commitment
**Proof Chain:** D → E
**Modules:**
```
mmsb-memory (produces D: AdmissionProof, E: CommitProof)
└── mmsb-storage (persistence)
```

### LearningBus
**Purpose:** Advisory derivation - pattern recognition
**Proof Chain:** F → G
**Modules:**
```
mmsb-learning (produces F: OutcomeProof)
└── mmsb-knowledge (produces G: KnowledgeProof)
```

### ResponseBus
**Purpose:** Outward-facing views - agent interface
**Flow:** StateQuery → StateProjection
**Modules:**
```
mmsb-memory (projects state views)
└── mmsb-response (formats agent responses)
```

### ComputeBus
**Purpose:** GPU/CUDA acceleration - parallel computation
**Flow:** ComputeRequest → GPU execution → Result
**Modules:**
```
mmsb-compute (GPU orchestration)
└── mmsb-kernel (CUDA kernels)
```

### ChromiumBus
**Purpose:** Browser automation - web interaction
**Flow:** BrowserCommand → Chromium → Result
**Modules:**
```
mmsb-chromium (browser control)
└── mmsb-web (web scraping/interaction)
```

### ReplayBus
**Purpose:** Observability - historical event stream
**Flow:** Historical events for replay/audit
**Modules:**
```
mmsb-memory (event source)
└── mmsb-replay (event replay engine)
```

---

## Bus Interaction Tree

```
┌─────────────────────────────────────────────────────────────┐
│                      MMSB Runtime                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  JudgmentBus                                               │
│  ┌──────────────────────────────────────────┐             │
│  │ Intent → Policy → Judgment               │             │
│  │   (A)      (B)       (C)                 │             │
│  └──────────────┬───────────────────────────┘             │
│                 │                                          │
│                 │ approved                                 │
│                 ▼                                          │
│  ExecutionBus                                              │
│  ┌──────────────────────────────────────────┐             │
│  │ Executor → Device → Propagation          │             │
│  └──────────────┬───────────────────────────┘             │
│                 │                                          │
│                 │ execution complete                       │
│                 ▼                                          │
│  StateBus                                                  │
│  ┌──────────────────────────────────────────┐             │
│  │ Memory (Admission D, Commit E)           │             │
│  │   ↓                                      │             │
│  │ Storage (persist)                        │             │
│  └──────────────┬───────────────────────────┘             │
│                 │                                          │
│                 ├────────────────────┐                     │
│                 │                    │                     │
│                 ▼                    ▼                     │
│  LearningBus           ResponseBus                         │
│  ┌──────────────┐     ┌──────────────────┐               │
│  │ Learning (F) │     │ Response         │               │
│  │   ↓          │     │   ↓              │               │
│  │ Knowledge(G) │     │ Agent interface  │               │
│  └──────────────┘     └──────────────────┘               │
│                                                            │
│  ComputeBus            ChromiumBus                         │
│  ┌──────────────┐     ┌──────────────────┐               │
│  │ Compute      │     │ Chromium         │               │
│  │   ↓          │     │   ↓              │               │
│  │ GPU/CUDA     │     │ Browser control  │               │
│  └──────────────┘     └──────────────────┘               │
│                                                            │
│  ReplayBus                                                 │
│  ┌──────────────────────────────────────────┐             │
│  │ Memory → Replay (historical stream)      │             │
│  └──────────────────────────────────────────┘             │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Bus Protocol Traits

Each bus defines a protocol trait that modules must implement:

### `JudgmentProtocol`
- `submit_intent() -> IntentProof`
- `evaluate_policy() -> PolicyProof`
- `exercise_judgment() -> JudgmentProof`

### `ExecutionProtocol`
- `request_execution() -> ExecutionRequest`
- `execute() -> ExecutionOutcome`

### `StateProtocol`
- `admit() -> AdmissionProof`
- `commit() -> CommitProof`

### `LearningProtocol`
- `observe_outcome() -> OutcomeProof`
- `derive_knowledge() -> KnowledgeProof`

### `ResponseProtocol`
- `query_state() -> StateQuery`
- `project_view() -> StateProjection`

### `ComputeProtocol`
- `offload_computation() -> ComputeRequest`
- `execute_kernel() -> ComputeResult`

### `ChromiumProtocol`
- `send_command() -> BrowserCommand`
- `execute_script() -> BrowserResult`

### `ReplayProtocol`
- `stream_events() -> EventStream`
- `replay_to_state() -> StateSnapshot`

---

## Module-Bus Mapping

| Module | Primary Bus | Secondary Buses |
|--------|-------------|-----------------|
| mmsb-intent | JudgmentBus | - |
| mmsb-policy | JudgmentBus | - |
| mmsb-judgment | JudgmentBus | ExecutionBus (triggers) |
| mmsb-executor | ExecutionBus | - |
| mmsb-device | ExecutionBus | - |
| mmsb-propagation | ExecutionBus | StateBus (writes) |
| mmsb-memory | StateBus | ResponseBus, ReplayBus |
| mmsb-storage | StateBus | - |
| mmsb-learning | LearningBus | - |
| mmsb-knowledge | LearningBus | ResponseBus |
| mmsb-response | ResponseBus | - |
| mmsb-compute | ComputeBus | - |
| mmsb-chromium | ChromiumBus | - |
| mmsb-replay | ReplayBus | - |

---

## Bus Isolation Properties

- **JudgmentBus:** Fast, high-priority, synchronous
- **ExecutionBus:** Slower, isolated failures don't block judgment
- **StateBus:** Serialized writes, concurrent reads
- **LearningBus:** Async, non-blocking, advisory only
- **ResponseBus:** Read-only, no authority
- **ComputeBus:** Async, GPU-bound workloads
- **ChromiumBus:** Async, network-bound operations
- **ReplayBus:** Read-only, historical data
