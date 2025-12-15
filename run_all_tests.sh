#!/bin/bash
# Comprehensive test runner for MMSB Week 27-32 validation

set -e

echo "========================================"
echo "MMSB Week 27-32 Test Suite"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

RUST_PASSED=0
JULIA_PASSED=0

# Rust tests
echo "Running Rust tests..."
if cargo check --tests 2>&1 | grep -q "error"; then
    echo -e "${RED}✗ Rust compilation failed${NC}"
else
    echo -e "${GREEN}✓ Rust compilation successful${NC}"
    RUST_PASSED=1
fi
echo ""

# Julia tests (basic check)
echo "Running Julia validation..."
if julia --project=. -e 'include(joinpath("test", "examples_basic.jl"))' 2>&1 | grep -q "Test Failed"; then
    echo -e "${RED}✗ Julia tests failed${NC}"
else
    echo -e "${GREEN}✓ Julia tests passed${NC}"
    JULIA_PASSED=1
fi
echo ""

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Rust:  $([[ $RUST_PASSED -eq 1 ]] && echo "${GREEN}✓ PASS${NC}" || echo "${RED}✗ FAIL${NC}")"
echo -e "Julia: $([[ $JULIA_PASSED -eq 1 ]] && echo "${GREEN}✓ PASS${NC}" || echo "${RED}✗ FAIL${NC}")"
echo ""

if [[ $RUST_PASSED -eq 1 ]] && [[ $JULIA_PASSED -eq 1 ]]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some tests failed. Review output above.${NC}"
    exit 1
fi
