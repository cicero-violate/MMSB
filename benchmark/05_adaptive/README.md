# Layer 05 — Adaptive Benchmarks

Path: `benchmark/05_adaptive/`

## Focus
- Layout optimizer convergence
- Adaptive page clustering heuristics

## Scenarios
1. `layout_optimizer_pass`
2. `page_clustering_kmeans`

## Running
```
julia --project=. benchmark/benchmarks.jl --group adaptive
```
Summary metrics are appended to `benchmark/results/phase6.json`.
