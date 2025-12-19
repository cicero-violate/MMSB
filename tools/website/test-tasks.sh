#!/bin/bash
BASE_URL="http://127.0.0.1:8888/mmsb"

echo "=== Testing Task System ==="

# Test 1: Create task
echo "Test 1: Create task"
curl -s "${BASE_URL}/create-task?task_id=TEST_TASK&description=Test%20workflow&_t=$(date +%s%3N)" | jq .

# Test 2: Add context
echo "Test 2: Add context"
curl -s "${BASE_URL}/add-context?task_id=TEST_TASK&query_url=/mmsb/test&_t=$(date +%s%3N)" | jq .

# Test 3: Add instruction
echo "Test 3: Add instruction"  
curl -s "${BASE_URL}/add-instruction?task_id=TEST_TASK&instruction_id=TEST_001&action=created&_t=$(date +%s%3N)" | jq .

# Test 4: Complete task
echo "Test 4: Complete task"
curl -s "${BASE_URL}/complete-task?task_id=TEST_TASK&status=completed&result=success&_t=$(date +%s%3N)" | jq .

# Test 5: List tasks
echo "Test 5: List tasks"
curl -s "${BASE_URL}/list-tasks?_t=$(date +%s%3N)" | jq .

# Test 6: Get task
echo "Test 6: Get task"
curl -s "${BASE_URL}/task?task_id=TEST_TASK&_t=$(date +%s%3N)" | jq .

# Cleanup
rm -rf /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tasks/TEST_TASK
echo '{}' > /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/tasks/index.json

echo "=== Tests Complete ==="
