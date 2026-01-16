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
cargo run --example 01_simple_rename
echo ""

echo "--- Example 2: Multi-File Refactor ---"
cargo run --example 02_multi_file_refactor
echo ""

echo "--- Example 3: Structural vs State ---"
cargo run --example 03_structural_vs_state
echo ""

echo "--- Example 4: Intent Extraction ---"
cargo run --example 04_intent_extraction
echo ""

echo "--- Example 5: Query Patterns ---"
cargo run --example 05_query_patterns
echo ""

echo "--- Example 6: Delta to Page ---"
cargo run --example 06_delta_to_page
echo ""

echo "--- Bridge Example ---"
cargo run --example bridge_example

echo ""
echo "=== All Examples Complete ==="
