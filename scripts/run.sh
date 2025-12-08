cargo build --release
RUST_BACKTRACE=full cargo run --example rust_smoke_test_ffi
RUST_BACKTRACE=full cargo run --example rust_smoke_replay_full
RUST_BACKTRACE=full cargo run --example rust_smoke_replay_full_cuda
RUST_BACKTRACE=full cargo run --example rust_smoke_checkpoint_roundtrip
