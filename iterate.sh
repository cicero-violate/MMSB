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
RUST_BACKTRACE=full RUSTFLAGS="-A warnings" cargo run --example rust_comprehensive_internal_tests
echo "--------------------------------------------------"
