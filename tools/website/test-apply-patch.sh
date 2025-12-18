#!/bin/bash
# Test script for apply-patch endpoint

BASE_URL="http://127.0.0.1:8888/mmsb"
TIMESTAMP=$(date +%s%3N)

echo "=== Testing Apply-Patch Endpoint ==="
echo ""

# Test 1: Simple patch
echo "Test 1: Apply simple patch to test file"
PATCH='*** Begin Patch
*** Add File: test_patch_output.txt
+This is a test file created by apply-patch endpoint
+Line 2
+Line 3
*** End Patch'

ENCODED=$(printf "%s" "$PATCH" | jq -sRr @uri)
RESPONSE=$(curl -s "${BASE_URL}/apply-patch?patch=${ENCODED}&_t=${TIMESTAMP}")
echo "$RESPONSE" | jq .
echo ""

# Test 2: Invalid patch format
echo "Test 2: Invalid patch (should fail)"
INVALID_PATCH='This is not a valid patch'
ENCODED=$(printf "%s" "$INVALID_PATCH" | jq -sRr @uri)
RESPONSE=$(curl -s "${BASE_URL}/apply-patch?patch=${ENCODED}&_t=$((TIMESTAMP+1))")
echo "$RESPONSE" | jq .
echo ""

# Test 3: Missing patch parameter
echo "Test 3: Missing patch parameter (should fail)"
RESPONSE=$(curl -s "${BASE_URL}/apply-patch?_t=$((TIMESTAMP+2))")
echo "$RESPONSE" | jq .
echo ""

# Test 4: Base64 encoding
echo "Test 4: Base64-encoded patch"
PATCH_BASE64=$(printf "%s" "$PATCH" | base64 -w 0)
RESPONSE=$(curl -s "${BASE_URL}/apply-patch?patch=${PATCH_BASE64}&encoding=base64&_t=$((TIMESTAMP+3))")
echo "$RESPONSE" | jq .
echo ""

# Cleanup
echo "Cleaning up test files..."
rm -f /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/test_patch_output.txt

echo "=== Tests Complete ==="
