#!/bin/bash
# Test instruction-builder endpoint

BASE_URL="http://127.0.0.1:8888/mmsb"

echo "=== Testing Build-Patch Endpoint ==="
echo ""

# Test 1: Simple replace with template
echo "Test 1: Template-based simple_replace"
curl -s "${BASE_URL}/build-patch?template=simple_replace&file=test_file.txt&old_line=old%20content&new_line=new%20content&context_before=line1&context_after=line2&_t=$(date +%s%3N)" | jq .
echo ""

# Test 2: Direct line replacement
echo "Test 2: Direct line replacement (no template)"
curl -s "${BASE_URL}/build-patch?file=test_direct.txt&line_before=before&line_after=after&context_1=ctx1&context_2=ctx2&context_3=ctx3&_t=$(date +%s%3N)" | jq .
echo ""

# Test 3: Missing parameters
echo "Test 3: Missing required parameters"
curl -s "${BASE_URL}/build-patch?_t=$(date +%s%3N)" | jq .
echo ""

# Test 4: Unknown template
echo "Test 4: Unknown template"
curl -s "${BASE_URL}/build-patch?template=nonexistent&file=test.txt&_t=$(date +%s%3N)" | jq .
echo ""

echo "=== Tests Complete ==="
