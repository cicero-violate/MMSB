# Benchmark Results

This directory captures all performance measurements emitted by the Phase 6 tooling.

| Artifact | Purpose |
|----------|---------|
| `baseline.json` | Legacy Julia benchmark suite captured prior to the Rust refactor. Useful for historical comparison. |
| `phase6.json` | Latest Rust Phase 6 run (`cargo run --bin phase6_bench --release`). Contains throughput and tick latency metrics. |

## Latest Run (`phase6.json`)
- Processed deltas: **20,000**
- Single-thread throughput: **1.8M deltas/sec**
- Multi-thread throughput (8 HW threads): **14.4M deltas/sec**
- Tick metrics:
  - Propagation: **10.3 ms**
  - Graph validation: **0.05 ms**
  - GC: **0 ms** (not triggered)
  - Total tick time: **11.9 ms** (budget < 16 ms ✅)

To refresh this file:
```bash
cargo run --bin phase6_bench --release
cat benchmark/results/phase6.json
```
