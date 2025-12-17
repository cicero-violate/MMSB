#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="-A warnings"
export RUST_BACKTRACE=full
export LD_LIBRARY_PATH="$(pwd)/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JULIA_PROJECT="$(pwd)"
export JULIA_FLAGS="--startup-file=no --project=."

run() {
  echo "--> $*"
  "$@"
  echo "--------------------------------------------------"
}

run cargo build --release
run cargo test --release --lib --tests
run cargo run --example rust_smoke_test_ffi
run cargo run --example rust_smoke_replay_full
run cargo run --example rust_smoke_checkpoint_roundtrip
# run cargo test --release --test stress_throughput -- --nocapture
run cargo test --release --test stress_stability -- --nocapture
run cargo test --release --test stress_memory -- --nocapture
run cargo run --bin phase6_bench --release
run julia $JULIA_FLAGS -e "using Pkg; Pkg.test()"
run julia $JULIA_FLAGS benchmark/benchmarks.jl
run julia $JULIA_FLAGS benchmark/validate_all.jl
