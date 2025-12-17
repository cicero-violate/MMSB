# Layer 04 — Propagation Benchmarks

Path: `benchmark/04_propagation/`

## Focus
- Lock-free ring buffer queue throughput
- ThroughputEngine delta processing
- TickOrchestrator latency (<16ms target)

## Scenarios
1. `ring_buffer_latency`
2. `delta_throughput_phase6`
3. `tick_latency_phase6`

## Running
```
cargo run --bin phase6_bench --release
```
The binary persists metrics to `benchmark/results/phase6.json` under `throughput` and `tick_latency`.
