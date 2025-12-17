#!/bin/bash
# Convenience script to run MMSB analyzer

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

echo "Checking mmsb-analyzer..."
cd "$SCRIPT_DIR"
cargo check

echo ""
echo "Running analysis on MMSB project..."
# cargo run -- \
#     --root "$PROJECT_ROOT" \
#     --output "$PROJECT_ROOT/docs/analysis" \
#     --julia-script "$SCRIPT_DIR/src/00_main.jl" \
#     "$@"

cargo run --release -- --skip-julia 

echo ""
echo "Analysis complete! Reports available in: $PROJECT_ROOT/docs/analysis/"
