#!/usr/bin/env bash
set -euo pipefail

export RUSTFLAGS="-A warnings"
export RUST_BACKTRACE=full

run() {
  echo "--> $*"
  "$@"
  echo "--------------------------------------------------"
}

run cargo build --release
run cargo run --example rust_smoke_test_ffi
run cargo run --example rust_smoke_replay_full
run cargo run --example rust_smoke_replay_full_cuda
run cargo run --example rust_smoke_checkpoint_roundtrip
run cargo test --example rust_comprehensive_internal_tests
run cargo test --lib --tests --no-default-features
run cargo run --bin phase6_bench --release
run julia --startup-file=no --project=. -e "using Pkg; Pkg.test();"
run julia --startup-file=no --project=. benchmark/benchmarks.jl
run julia --startup-file=no --project=. test/week27_31_integration.jl
