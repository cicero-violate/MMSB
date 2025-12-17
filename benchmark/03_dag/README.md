# Layer 03 — DAG Benchmarks

Path: `benchmark/03_dag/`

## Focus
- ShadowPageGraph traversal + validation
- Cycle detection and topological sort performance

## Scenarios
1. `graph_validator_cycle_scan`
2. `topological_sort`
3. `bfs_descendants`

## Running
```
julia --project=. benchmark/benchmarks.jl --group dag
```
Results append to `benchmark/results/phase6.json`.
