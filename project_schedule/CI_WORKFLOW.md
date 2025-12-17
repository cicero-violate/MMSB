# MMSB CI Workflow Plan

**Updated:** 2025-12-17  
**Status:** Draft – feeds T7.8 task in `project_schedule/05_DAG_DEPENDENCIES.md`

---

## Goals
- Run Rust + Julia validation on every push/PR.
- Publish benchmark artifacts (`benchmark/results/*.json`, validator summaries).
- Keep wall-clock under 25 minutes by splitting jobs and caching toolchains.

---

## Pipeline Overview
| Job | Runner | Steps | Artifacts |
|-----|--------|-------|-----------|
| `lint` | ubuntu-latest | `cargo fmt -- --check`, `cargo clippy -- -D warnings` | — |
| `rust-tests` | ubuntu-latest | `cargo test --release --lib --tests`, `cargo test --release --test stress_* -- --nocapture` | `target/release` test logs |
| `julia-bench` | ubuntu-latest | Install Julia, `benchmark/benchmarks.jl`, `benchmark/validate_all.jl` | `benchmark/results/julia_phase6.json`, validation log |
| `phase6-bench` | ubuntu-latest | `cargo run --bin phase6_bench --release` | `benchmark/results/phase6.json` |

Jobs share a cargo cache keyed on `Cargo.lock` + toolchain hash. Julia job caches `~/.julia/artifacts`.

---

## Required Environment
```yaml
env:
  RUSTFLAGS: "-A warnings"
  RUST_BACKTRACE: full
  LD_LIBRARY_PATH: "${{ github.workspace }}/target/release:${LD_LIBRARY_PATH}"
  JULIA_PROJECT: "${{ github.workspace }}"
```

Each Julia step uses `julia --startup-file=no --project=. ...`.

---

## Workflow Skeleton
```yaml
name: Benchmarks
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 6 * * 1"  # weekly sanity run

jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --release --lib --tests
      - run: cargo test --release --test stress_throughput -- --nocapture
      - run: cargo test --release --test stress_stability -- --nocapture
      - run: cargo test --release --test stress_memory -- --nocapture

  julia-bench:
    needs: rust-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.10'
      - uses: julia-actions/cache@v1
      - run: julia --startup-file=no --project=. benchmark/benchmarks.jl
      - run: julia --startup-file=no --project=. benchmark/validate_all.jl | tee validation.log
      - uses: actions/upload-artifact@v4
        with:
          name: julia-validation
          path: |
            validation.log
            benchmark/results/julia_phase6.json
```

`phase6-bench` mirrors the Julia job but runs `cargo run --bin phase6_bench --release` and uploads `benchmark/results/phase6.json`.

---

## Enforcement
- Configure branch protection to require `rust-tests`, `julia-bench`, and `phase6-bench`.
- Keep `lint` optional but blocking for formatting errors.
- Weekly scheduled run ensures nightlies stay green even without PR traffic.

---

## Follow-ups
- Add slack/webhook notifications after workflow stabilizes.
- Extend matrix to include CUDA runner when hardware is available (Phase 8).
