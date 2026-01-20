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

echo "--- Example 1: Test Schema ---"
cargo run --example test_schema
echo ""

echo ""
echo "=== All Examples Complete ==="
