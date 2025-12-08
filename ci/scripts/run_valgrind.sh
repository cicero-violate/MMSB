#!/usr/bin/env bash
set -euo pipefail
if ! command -v valgrind >/dev/null; then
  echo "valgrind not available; skipping"
  exit 0
fi
cargo test --no-run
for testbin in target/debug/deps/mmsb_*; do
  if [[ -x "$testbin" ]]; then
      valgrind --error-exitcode=1 "$testbin"
  fi
done
