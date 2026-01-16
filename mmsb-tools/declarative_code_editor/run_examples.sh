#!/bin/bash
set -e

echo "=== Building declarative_code_editor ==="
cargo build

echo ""
echo "=== Running tests ==="
cargo test --lib

echo ""
echo "=== Running tests ==="
cargo test 

echo ""
echo "=== Running bridge example ==="
cargo run --example bridge_example
