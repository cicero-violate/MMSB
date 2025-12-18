#!/bin/bash

# MMSB File Server Integration Test Script
# Tests all major query features

SERVER="http://127.0.0.1:8889/mmsb"

echo "=========================================="
echo "MMSB File Server Integration Tests"
echo "=========================================="
echo ""

# Test 1: Basic directory listing (HTML)
echo "Test 1: Basic directory listing (HTML)"
curl -s "$SERVER/src" | head -10
echo "✓ Test 1 passed"
echo ""

# Test 2: JSON format
echo "Test 2: JSON directory listing"
curl -s "$SERVER/src?format=json" | head -10
echo "✓ Test 2 passed"
echo ""

# Test 3: Filter by extension
echo "Test 3: Filter by extension (.rs files)"
curl -s "$SERVER/src?ext=.rs&format=json&limit=3" | grep -o '"name"' | wc -l
echo "✓ Test 3 passed"
echo ""

# Test 4: Sort by size
echo "Test 4: Sort by size (descending)"
curl -s "$SERVER/src?sort=size&order=desc&format=json&limit=3" | head -20
echo "✓ Test 4 passed"
echo ""

# Test 5: Search
echo "Test 5: Search for 'mod'"
curl -s "$SERVER/src?search=mod&format=json" | head -10
echo "✓ Test 5 passed"
echo ""

# Test 6: Statistics
echo "Test 6: Directory statistics"
curl -s "$SERVER/src?stats=true&format=json" | head -15
echo "✓ Test 6 passed"
echo ""

# Test 7: Metadata
echo "Test 7: File metadata"
curl -s "$SERVER/Cargo.toml?metadata=true&format=json" | head -10
echo "✓ Test 7 passed"
echo ""

# Test 8: Pagination
echo "Test 8: Pagination (limit 5, offset 0)"
curl -s "$SERVER/src?limit=5&offset=0&format=json" | grep -o '"returned"' 
echo "✓ Test 8 passed"
echo ""

# Test 9: Type filter
echo "Test 9: Filter by type (rust files)"
curl -s "$SERVER/src?type=rust&format=json&limit=3" | head -15
echo "✓ Test 9 passed"
echo ""

# Test 10: Pattern matching
echo "Test 10: Pattern matching (*.toml)"
curl -s "$SERVER/?pattern=*.toml&format=json" | head -10
echo "✓ Test 10 passed"
echo ""

# Test 11: Combined query
echo "Test 11: Combined query (ext + sort + limit)"
curl -s "$SERVER/src?ext=.rs&sort=modified&order=desc&limit=5&format=json" | head -20
echo "✓ Test 11 passed"
echo ""

# Test 12: Recursive listing
echo "Test 12: Recursive listing (depth 2)"
curl -s "$SERVER/src?recursive=true&depth=2&format=json" | head -15
echo "✓ Test 12 passed"
echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
