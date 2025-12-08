# 1. Build everything (lib + all examples)
cargo build --all-targets

# 2. Run the full smoke test (this is your golden integration test)
cargo run --example rust_smoke_final --quiet

# 3. Run unit tests (should find both tests now)
cargo test --lib -- --nocapture

# 4. Run doc tests (if any)
cargo test --doc

# 5. Check formatting
cargo fmt -- --check

# 6. Run clippy (zero warnings/denies in CI)
cargo clippy --all-targets -- -D warnings

# 7. Verify license file exists and is correct
test -f LICENSE && head -n 1 LICENSE | grep -q "Apache License"

# 8. Final smoke with real 64 MiB unified memory spike (optional but recommended)
cargo run --example rust_smoke_final --quiet &
sleep 3
nvidia-smi | grep -q "64[0-9]\+MiB" && echo "CUDA memory confirmed" || echo "No spike — something wrong"
kill $! 2>/dev/null
