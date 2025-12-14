# Intent Replay with Metadata

Intent metadata persists alongside every delta written to the MMSB transaction log. Phase 5 adds encode/decode plumbing across the Rust FFI so Julia can tag deltas with structured metadata and recover that information during replay.

## Recording Intent Metadata

1. Build an `UpsertPlan` describing the query, predicate, and delta payload.
2. Call `IntentLowering.execute_upsert_plan!(state, plan; source=:intent)` to evaluate the predicate, construct the delta, and apply it through the canonical router.
3. MMSB automatically augments the provided plan metadata with the query string and the assigned `delta_id`, then persists the metadata JSON inside the TLog frame.

```julia
plan = MMSB.UpsertPlan.UpsertPlan(
    "select hot pages",
    bytes -> sum(bytes) == 0,
    MMSB.UpsertPlan.DeltaSpec(UInt64(page.id), payload, trues(length(payload))),
    Dict(:intent_id => "hot-path", :author => "planner"),
)
IntentLowering.execute_upsert_plan!(state, plan)
```

## Filtering during Replay

Every replay entry exposes structured metadata via `MMSB.DeltaTypes.intent_metadata(delta; parse=true)`. Combine this helper with `ReplayEngine.replay_with_predicate` to select specific intents:

```julia
filtered = MMSB.ReplayEngine.replay_with_predicate(state) do epoch, delta
    meta = MMSB.DeltaTypes.intent_metadata(delta; parse=true)
    meta !== nothing && get(meta, :intent_id, nothing) == "hot-path"
end
```

Use the filtered tuples with `ReplayEngine.incremental_replay!` to reconstruct the state implied by the selected intents.

## Checkpoint Verification

`TLog.checkpoint_log!` persists the allocator, page data, and the enriched log frames. Loading the checkpoint with `TLog.load_checkpoint!` restores both page data and intent metadata, enabling deterministic replay after a restart.

```julia
mktemp() do path, io
    close(io)
    MMSB.TLog.checkpoint_log!(state, path)
    restored = MMSB.MMSBStateTypes.MMSBState(MMSB.MMSBStateTypes.MMSBConfig(tlog_path=path))
    MMSB.TLog.load_checkpoint!(restored, path)
    @show MMSB.DeltaTypes.intent_metadata(MMSB.TLog.query_log(restored)[end]; parse=true)
end
```

This workflow satisfies INT.G2 deliverables: intent history can be reconstructed, filtered, and validated across checkpoints using the same deterministic pipeline that backs the core replay engine.
