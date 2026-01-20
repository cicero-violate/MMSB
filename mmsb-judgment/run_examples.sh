#!/bin/bash
set -e

echo "=== Building declarative_code_editor ==="
cargo build

echo ""
echo "=== Running Tests ==="
cargo test --lib

echo ""
echo "=== Running Examples ==="
echo ""

echo "--- Example 1: Test Policy ---"
cargo run --example issue_cli
echo ""

echo ""
echo "=== All Examples Complete ==="
