# MMSB Query / Mutate / Upsert (QMU) Guide

The QMU model partitions every interaction with MMSB into three pure modes:

| Mode   | Layer(s) | Language | Responsibility                                  |
|--------|----------|----------|-------------------------------------------------|
| Query  | L1 + L3  | Rust + Julia wrappers | Pure reads across pages, DAGs, and logs. |
| Mutate | L1       | Rust     | Apply canonical deltas, account in the TLog.    |
| Upsert | L7       | Julia    | Build conditional plans that *produce* deltas.  |

No operation is allowed to straddle modes: Queries never write, Mutations never reason, and Upserts never execute. This guide documents the public API surface that implements these guarantees along with end-to-end examples and the canonical intent → delta flow.

---

## Query Operations (Read-Only)

Query helpers are thin Julia wrappers around Rust code that expose *pure* inspection APIs. They never mutate state and can be replayed deterministically.

| Helper                                       | Path                    | Description                                            |
| ---                                          | ---                     | ---                                                    |
| `API.query_page(state, page_id)`             | `src/API.jl`            | Returns a copy of the page bytes (L1 `read_page`).     |
| `TLog.query_log(state; filters…)`            | `src/01_page/TLog.jl`   | Streams canonical deltas + metadata for observability. |
| `ReplayEngine.replay_to_epoch(state, epoch)` | `src/01_page/replay.jl` | Reconstructs state purely from persisted deltas.       |
| `ShadowGraph.get_children(graph, page_id)`   | `src/03_dag`            | DAG traversal helpers for reasoning/planning layers.   |

**Semantics**

1. Queries always copy data out of MMSB-managed pages. No aliasing back into execution buffers is permitted (`API.query_page` allocates a fresh `Vector{UInt8}`).
2. Queries may filter deltas or compute statistics, but they cannot enqueue new deltas or mutate allocator state.
3. Query-heavy workflows (debuggers, replay inspectors) remain deterministic because they read the same canonical deltas that back replay.

---

## Mutate Operations (Delta Application)

Mutate helpers accept canonical deltas and execute them strictly inside Rust (L1). Julia never writes bytes directly; it requests deltas that Rust validates, logs, and propagates.

| Helper                                                                    | Path                          | Description                                                                                   |
| ---                                                                       | ---                           | ---                                                                                           |
| `API.update_page(state, page_id, bytes; source=:api)`                     | `src/API.jl`                  | Convenience wrapper that diffs the current page, builds a dense delta, and enqueues it.       |
| `DeltaRouter.create_delta(state, page_id, mask, payload; source=:router)` | `src/01_page/delta_router.rs` | Constructs sparse/dense deltas with explicit masks.                                           |
| `DeltaRouter.route_delta!(state, delta; propagate=true)`                  | same                          | Validates deltas (`delta_validation.rs`), records them in the TLog, and notifies propagation. |
| `DeltaRouter.batch_route_deltas!(state, deltas)`                          | same                          | Batches deltas before scheduling propagation.                                                 |

**Semantics**

1. Every mutation must pass through `validate_delta()` (Rust). Payloads with length mismatches, invalid masks, or inconsistent sources are rejected before touching state.
2. Mutation execution logs intent metadata in the TLog (see Upsert operations). Replay re-applies these deltas byte-for-byte—there is no shortcut path.
3. Mutations are deterministic and idempotent: re-routing the same delta yields the same state thanks to canonicalization inside `Delta::apply_to`.

---

## Upsert Operations (Conditional Writes)

Upserts live entirely in Julia (L7). They are *plans* that describe what should happen if a predicate holds; they never mutate state directly.

| Helper                                                             | Path                                  | Description                                                                                                          |
| ---                                                                | ---                                   | ---                                                                                                                  |
| `UpsertPlan.UpsertPlan`                                            | `src/07_intention/UpsertPlan.jl`      | Pure struct containing `query::String`, `predicate`, `DeltaSpec`, `metadata`.                                        |
| `IntentLowering.lower_intent_to_deltaspec(plan)`                   | `src/07_intention/intent_lowering.jl` | Converts a plan into an FFI-friendly description (`payload`, `mask_bytes`, metadata).                                |
| `IntentLowering.execute_upsert_plan!(state, plan; source=:intent)` | same                                  | Evaluates the predicate via a **query snapshot**, creates a delta if allowed, and routes it through the mutate path. |

**Semantics**

1. Plans declare *intent* but defer execution to the mutate pipeline. If the predicate fails, nothing is logged or mutated—callers receive `(applied=false, query_snapshot=…)`.
2. Metadata is merged with the plan’s query string and persisted via `DeltaTypes.set_intent_metadata!`, enabling deterministic replay and audit trails.
3. Upsert producers (reasoning/planning layers or external agents) can safely run offline: plans are serializable, deterministic, and validated (`validate_plan`) before execution.

---

## Usage Example

```julia
using MMSB
using MMSB.API
using MMSB.IntentLowering
using MMSB.UpsertPlan

state = mmsb_start(enable_gpu=false)
page = create_page(state; size=4, metadata=Dict(:role => :counter))

# --- Mutate (M)
API.update_page(state, page.id, UInt8[0x00, 0x00, 0x00, 0x01]; source=:boot)

# --- Query (Q)
baseline = API.query_page(state, page.id)
@assert baseline == UInt8[0x00, 0x00, 0x00, 0x01]

# --- Upsert (U)
mask = trues(length(baseline))
plan = UpsertPlan.UpsertPlan(
    "select counter from page $(Int(page.id))",
    data -> sum(data) < 5,
    UpsertPlan.DeltaSpec(UInt64(page.id), UInt8[0x00, 0x00, 0x00, 0x02], mask),
    Dict(:intent_id => "increment")
)
result = IntentLowering.execute_upsert_plan!(state, plan)
@assert result.applied
@assert API.query_page(state, page.id) == plan.deltaspec.payload

mmsb_stop(state)
```

The snippet touches all three modes:

1. **Mutate** updates the page deterministically via canonical deltas.
2. **Query** reads state snapshots without mutating anything.
3. **Upsert** captures conditional intent, verifies it via a read-only snapshot, and—only if allowed—hands control back to Mutate.

---

## Architecture Diagram

```
Intent (L8+/external)
        │  author intent, choose UpsertPlan template
        ▼
UpsertPlan (L7 Julia: query, predicate, DeltaSpec, metadata)
        │  validate_plan / predicate(snapshot)
        ▼
DeltaSpec (L7 Julia) ── lower_intent_to_deltaspec ──▶ FFI Bridge (L7↔L1)
        │                                               │
        │                                 validate_delta (L1 Rust)
        ▼                                               │
Delta (L1 Rust canonical form) ──▶ TLog (persist intent metadata)
        │
        ▼
Propagation Engine (L4 Rust) ──▶ Dependent layers (L5+)
```

This “Intent → Delta” pipeline ensures:

* **Rust executes only** (`DeltaRouter`, `Propagation`).
* **Julia reasons only** (`UpsertPlan`, predicates).
* **All writes use canonical deltas** (validated before execution).
* **Intent metadata stays replayable** (`TLog`, `ReplayEngine`).

See `test/test_week24_25_integration.jl` for exhaustive QMU separation tests that back this documentation.

---

## Further Reading

* `docs/API.md` – full public API reference.
* `docs/intent_replay.md` – replay workflows leveraging intent metadata.
* `project_schedule/CLAUDE.md` – non-negotiable architectural rules behind QMU.
