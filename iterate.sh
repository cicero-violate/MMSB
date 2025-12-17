echo "Building project in release mode..."
RUSTFLAGS="-A warnings" cargo build --release
echo "--------------------------------------------------"

echo "Running rust_smoke_test_ffi..."
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo run --example rust_smoke_test_ffi
echo "--------------------------------------------------"

echo "Running rust_smoke_replay_full..."
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo run --example rust_smoke_replay_full
echo "--------------------------------------------------"

echo "Running rust_smoke_replay_full_cuda..."
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo run --example rust_smoke_replay_full_cuda
echo "--------------------------------------------------"

echo "Running rust_smoke_checkpoint_roundtrip..."
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo run --example rust_smoke_checkpoint_roundtrip
echo "--------------------------------------------------"

echo "Running rust_smoke_checkpoint_roundtrip..."
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo test --example rust_comprehensive_internal_tests
echo "--------------------------------------------------"

echo "Running cargo test"
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo test 
echo "--------------------------------------------------"

echo "Running Julia tests.."
julia --startup-file=no --project=. -e "using Pkg; Pkg.test();"
echo "--------------------------------------------------"

echo "Running Julia tests.."
julia --startup-file=no --project=. benchmark/benchmarks.jl
echo "--------------------------------------------------"

echo "Running Julia tests.."
julia --startup-file=no --project=. test/week27_31_integration.jl
echo "--------------------------------------------------"

cargo run --bin phase6_bench --release
cargo test --lib --no-default-features
