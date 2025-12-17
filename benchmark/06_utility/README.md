# Layer 06 — Utility Benchmarks

Path: `benchmark/06_utility/`

## Focus
- Telemetry aggregation cost
- Invariant checker batch runs
- Memory monitor snapshots/GC triggers

## Scenarios
1. `telemetry_snapshot`
2. `invariant_batch`
3. `memory_monitor_gc`

## Running
```
julia --project=. benchmark/benchmarks.jl --group utility
```
Artifacts are merged into `benchmark/results/phase6.json`.
