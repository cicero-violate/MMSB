# Layer 00 — Physical Benchmarks

Path: `benchmark/00_physical/`

## Focus
- Unified memory allocation and release latency
- GPU ↔ CPU data migration and NCCL setup costs
- Device synchronization overhead captured by `DeviceSync`

## Scenarios
1. `allocator_latency` — allocate/free varying page sizes (1KB → 1MB)
2. `unified_migration` — measure page migration via `UnifiedMemory`
3. `gpu_kernel_launch` — stress CUDA launch latency (stubbed in CPU env)

## Running
Execute:
```
julia --project=. benchmark/benchmarks.jl --group physical
```
Results are written to `benchmark/results/phase6.json` and summarized in the schedule.
