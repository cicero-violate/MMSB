# Layer 01 — Page Benchmarks

Path: `benchmark/01_page/`

## Focus
- Delta creation/merge throughput (dense + sparse)
- TLog append/replay costs
- Checkpoint write/restore latency

## Scenarios
1. `delta_dense_vs_sparse`
2. `tlog_append_replay`
3. `checkpoint_roundtrip`

## Running
```
julia --project=. benchmark/benchmarks.jl --group page
```
Aggregated metrics go to `benchmark/results/phase6.json`.
