# Layer 02 — Semiring Benchmarks

Path: `benchmark/02_semiring/`

## Focus
- Boolean/Tropical semiring operations (`fold_add`, `fold_mul`, `accumulate`)
- Purity validator workloads

## Scenarios
1. `boolean_semiring_suite`
2. `tropical_semiring_suite`
3. `purity_validator`

## Running
```
julia --project=. benchmark/benchmarks.jl --group semiring
```
Outputs merge into `benchmark/results/phase6.json`.
