#!/bin/bash
# Test script to verify all the fixes for TO_DO_FIX.md issues

SERVER="https://cheese-server.duckdns.org/mmsb"
FAIL=0

pass() { echo "✓ PASS"; }
fail() { echo "✗ FAIL: $1"; FAIL=1; }

test_feature() {
    NAME="$1"
    URL="$2"
    EXPECT_CODE="${3:-200}"
    EXPECT_TEXT="$4"
    
    echo -n "Testing: $NAME ... "
    
    RESP=$(curl -s -w "\n%{http_code}" "$URL" 2>/dev/null)
    BODY=$(echo "$RESP" | sed '$d')
    CODE=$(echo "$RESP" | tail -n1)
    
    if [[ "$CODE" != "$EXPECT_CODE" ]]; then
        fail "Expected HTTP $EXPECT_CODE, got $CODE"
        return 1
    fi
    
    if [[ -n "$EXPECT_TEXT" ]] && ! echo "$BODY" | grep -qi "$EXPECT_TEXT"; then
        fail "Missing expected text: $EXPECT_TEXT"
        echo "Response (first 300 chars):"
        echo "$BODY" | head -c 300
        echo -e "\n"
        return 1
    fi
    
    pass
    return 0
}

echo "============================================================"
echo "MMSB File Server - Testing Fixed Features"
echo "============================================================"
echo ""

echo "=== 1. SORTING TESTS ==="
test_feature "Sort by name (asc)" \
    "$SERVER/src?sort=name&order=asc&format=json" \
    200 \
    '"entries"'

test_feature "Sort by size (desc)" \
    "$SERVER/src?sort=size&order=desc&format=json" \
    200 \
    '"size"'

test_feature "Sort by modified (desc)" \
    "$SERVER/src?sort=modified&order=desc&format=json" \
    200 \
    '"modified"'

test_feature "Sort by type" \
    "$SERVER/src?sort=type&order=asc&format=json" \
    200 \
    '"entries"'

echo ""
echo "=== 2. STATISTICS TESTS ==="
test_feature "Stats without format param (auto JSON)" \
    "$SERVER/src?stats=true" \
    200 \
    '"total_files"'

test_feature "Stats with explicit format=json" \
    "$SERVER/src?stats=true&format=json" \
    200 \
    '"total_dirs"'

echo ""
echo "=== 3. METADATA TESTS ==="
test_feature "Metadata without format param (auto JSON)" \
    "$SERVER/Cargo.toml?metadata=true" \
    200 \
    '"size"'

test_feature "Metadata with explicit format=json" \
    "$SERVER/Cargo.toml?metadata=true&format=json" \
    200 \
    '"modified"'

test_feature "Metadata with created timestamp" \
    "$SERVER/README.md?metadata=true" \
    200 \
    '"created"'

echo ""
echo "=== 4. PAGINATION TESTS ==="
test_feature "Pagination with limit" \
    "$SERVER/src?limit=5&format=json" \
    200 \
    '"returned"'

test_feature "Pagination with limit and offset" \
    "$SERVER/src?limit=3&offset=5&format=json" \
    200 \
    '"offset"'

test_feature "Page-based pagination" \
    "$SERVER/src?page=2&pagesize=5&format=json" \
    200 \
    '"pagination"'

echo ""
echo "=== 5. PRETTY PRINT TESTS ==="
test_feature "Pretty JSON output" \
    "$SERVER/src?format=json&pretty=true&limit=2" \
    200 \
    $'{\n  "'

test_feature "Pretty with metadata" \
    "$SERVER/Cargo.toml?metadata=true&pretty=true" \
    200 \
    $'{\n  "name"'

echo ""
echo "=== 6. TYPE FILTERING TESTS ==="
test_feature "Filter by type: rust" \
    "$SERVER/src?type=rust&format=json" \
    200 \
    '".rs"'

test_feature "Filter by type: config" \
    "$SERVER/?type=config&format=json" \
    200 \
    '"entries"'

test_feature "Filter by type: markdown" \
    "$SERVER/?type=markdown&format=json" \
    200 \
    '".md"'

echo ""
echo "=== 7. RECURSIVE + JSON TESTS ==="
test_feature "Recursive listing with JSON format" \
    "$SERVER/examples?recursive=true&format=json&depth=2" \
    200 \
    '"tree"'

test_feature "Recursive with stats in JSON" \
    "$SERVER/ci?recursive=true&format=json&depth=1" \
    200 \
    '"stats"'

echo ""
echo "=== 8. COMBINED FEATURE TESTS ==="
test_feature "Sort + Pagination + Pretty" \
    "$SERVER/src?sort=modified&order=desc&limit=3&format=json&pretty=true" \
    200 \
    '"modified"'

test_feature "Type filter + Sort + Pagination" \
    "$SERVER/?type=rust&sort=size&order=desc&limit=5&format=json" \
    200 \
    '".rs"'

test_feature "Extension + Sort + Pretty" \
    "$SERVER/src?ext=.rs&sort=name&format=json&pretty=true&limit=3" \
    200 \
    $'{\n  "'

echo ""
echo "============================================================"

if [[ $FAIL -eq 0 ]]; then
    echo "✓ ALL TESTS PASSED - All fixes working correctly!"
    echo "============================================================"
    exit 0
else
    echo "✗ SOME TESTS FAILED - Please review the output above"
    echo "============================================================"
    exit 1
fi
