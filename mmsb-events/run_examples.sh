#!/bin/bash
set -e

echo "=== Building declarative_code_editor ==="
cargo build

echo ""
echo "=== Running Tests ==="
# cargo test --lib

echo ""
echo "=== Running Examples ==="
echo ""

echo "--- Example 1: events_with_proof ---"
cargo run --example proofs_with_events
echo ""

echo ""
echo "=== All Examples Complete ==="
