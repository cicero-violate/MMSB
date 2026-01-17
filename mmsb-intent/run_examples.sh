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

echo "--- Example 1: Simple Rename ---"
cargo run --example classify_intent
echo ""

echo ""
echo "=== All Examples Complete ==="
