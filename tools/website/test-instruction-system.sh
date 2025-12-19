#!/bin/bash
# Test complete instruction system: save → list → apply

BASE_URL="http://127.0.0.1:8888/mmsb"
WORKDIR="/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB"

echo "=== Testing Instruction System ==="
echo ""

# Setup
echo "Creating test file..."
cat > "$WORKDIR/test_instruction.txt" << 'EOF'
header
old value
footer
EOF

# Test 1: Save instruction
echo "Test 1: Save instruction"
curl -s "${BASE_URL}/save-instruction?id=TEST_001&file=test_instruction.txt&template=simple_replace&old_line=old%20value&new_line=new%20value&context_before=header&context_after=footer&purpose=Test%20instruction&risk=low&_t=$(date +%s%3N)" | jq .
echo ""

# Test 2: List instructions
echo "Test 2: List instructions"
curl -s "${BASE_URL}/list-instructions?_t=$(date +%s%3N)" | jq .
echo ""

# Test 3: Get instruction details
echo "Test 3: Get instruction TEST_001"
curl -s "${BASE_URL}/instruction?id=TEST_001&_t=$(date +%s%3N)" | jq .
echo ""

# Test 4: Apply instruction
echo "Test 4: Apply instruction TEST_001"
curl -s "${BASE_URL}/apply-instruction?id=TEST_001&_t=$(date +%s%3N)" | jq .
echo ""

# Verify file changed
echo "Verifying file contents:"
cat "$WORKDIR/test_instruction.txt"
echo ""

# Test 5: Duplicate ID (should fail)
echo "Test 5: Try to save duplicate ID"
curl -s "${BASE_URL}/save-instruction?id=TEST_001&file=dummy.txt&template=simple_replace&old_line=x&new_line=y&_t=$(date +%s%3N)" | jq .
echo ""

# Test 6: Apply non-existent instruction
echo "Test 6: Apply non-existent instruction"
curl -s "${BASE_URL}/apply-instruction?id=NONEXISTENT&_t=$(date +%s%3N)" | jq .
echo ""

# Cleanup
echo "Cleaning up..."
rm -f "$WORKDIR/test_instruction.txt"
rm -rf "$WORKDIR/instructions/TEST_001"

# Rebuild index without TEST_001
cat > "$WORKDIR/instructions/index.json" << 'EOF'
{}
EOF

echo "=== Tests Complete ==="
