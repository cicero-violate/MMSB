#!/usr/bin/env bash
set -euo pipefail
if ! command -v rustup >/dev/null; then
  echo "rustup not found; skipping miri run"
  exit 0
fi
rustup toolchain install nightly --no-self-update
rustup component add miri --toolchain nightly
cargo +nightly miri setup
cargo +nightly miri test --lib
