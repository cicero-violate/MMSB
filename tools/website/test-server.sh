#!/bin/bash

SERVER="https://cheese-server.duckdns.org/mmsb"
FAIL=0

pass() { echo "✓ PASS"; }
fail() { echo "✗ FAIL"; FAIL=1; }

run_test () {
    NAME="$1"
    URL="$2"
    EXPECT="$3"

    echo "$NAME"

    RESP=$(curl -s -w "\n%{http_code}" "$URL")
    BODY=$(echo "$RESP" | sed '$d')
    CODE=$(echo "$RESP" | tail -n1)

    if [[ "$CODE" != "200" ]]; then
        echo "HTTP $CODE"
        fail
        return
    fi

    if [[ -n "$EXPECT" ]] && ! echo "$BODY" | grep -q "$EXPECT"; then
        echo "Missing expected token: $EXPECT"
        echo "Response body (first 200 chars):"
        echo "$BODY" | head -c 200
        echo ""
        fail
        return
    fi

    pass
    echo
}

echo "=========================================="
echo "MMSB File Server Integration Tests"
echo "=========================================="
echo

run_test "Test 1: Basic directory listing (HTML)" \
    "$SERVER/src" \
    "<html"

run_test "Test 2: JSON directory listing" \
    "$SERVER/src?format=json" \
    "\"entries\""

run_test "Test 3: Filter by extension (.rs)" \
    "$SERVER/src?ext=.rs&format=json&limit=3" \
    "\"name\""

run_test "Test 4: Sort by size (desc)" \
    "$SERVER/src?sort=size&order=desc&format=json&limit=3" \
    "\"entries\""

run_test "Test 5: Search for 'mod' in subdirectory" \
    "$SERVER/src/00_physical?search=mod&format=json" \
    "mod"

run_test "Test 6: Directory statistics" \
    "$SERVER/src?stats=true&format=json" \
    "\"total_files\""

run_test "Test 7: File metadata" \
    "$SERVER/Cargo.toml?metadata=true&format=json" \
    "\"modified\""

run_test "Test 8: Pagination" \
    "$SERVER/src?limit=5&offset=0&format=json" \
    "\"returned\""

run_test "Test 9: Type filter (rust)" \
    "$SERVER/src?type=rust&format=json&limit=15" \
    ".rs"

run_test "Test 10: Pattern matching (*.toml)" \
    "$SERVER/?pattern=*.toml&format=json" \
    ".toml"

run_test "Test 11: Combined query" \
    "$SERVER/src?ext=.rs&sort=modified&order=desc&limit=5&format=json" \
    "\"modified\""

run_test "Test 12: Recursive listing" \
    "$SERVER/src?recursive=true&depth=2&format=json" \
    "\"path\""

echo "=========================================="

if [[ $FAIL -eq 0 ]]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
    exit 1
fi

echo "=========================================="
