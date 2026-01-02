#!/bin/bash
# Run MMSB executor against correction intelligence output
# Generic version - works from any MMSB tool project

set -euo pipefail

# Auto-detect directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
EXEC_DIR="$(cd "$SCRIPT_DIR/../mmsb-executor" && pwd)"
OUTPUT_DIR="$ROOT_DIR/docs/97_correction_intelligence/slice_path_coherence"
TIMEOUT_SECONDS=1200
ADMISSION_ARTIFACT="$ROOT_DIR/admission_composition.json"
EXPECTED_SCHEMA_VERSION="0.1.0"

# ============================================================================
# PHASE 6.5 EXECUTION PRECONDITION (CATEGORY 1 WIRING)
# ============================================================================
# Execution MUST NOT proceed without admissible batch.
# This is a hard gate. No retries. No fallbacks. No flags.
# ============================================================================

# Check 1: Artifact exists
if [ ! -f "$ADMISSION_ARTIFACT" ]; then
    echo "Execution blocked: admission artifact not found" >&2
    echo "Expected: $ADMISSION_ARTIFACT" >&2
    exit 1
fi

# Check 2: Batch is admissible
ADMISSIBLE=$(jq -r '.admissible' "$ADMISSION_ARTIFACT")
if [ "$ADMISSIBLE" != "true" ]; then
    echo "Execution blocked: batch is not admissible (see admission_composition.json)" >&2
    exit 1
fi

# Check 3: Schema version matches
ARTIFACT_VERSION=$(jq -r '.schema_version' "$ADMISSION_ARTIFACT")
if [ "$ARTIFACT_VERSION" != "$EXPECTED_SCHEMA_VERSION" ]; then
    echo "Execution blocked: schema version mismatch" >&2
    echo "Expected: $EXPECTED_SCHEMA_VERSION, Got: $ARTIFACT_VERSION" >&2
    exit 1
fi

# Precondition satisfied - proceeding with execution
echo "✅ Admission precondition satisfied (admissible batch, schema v$ARTIFACT_VERSION)"

# ============================================================================

cd "$EXEC_DIR"

# Pick one (uncomment the block you want):
# 1) Dry run (no mutations)
# timeout "$TIMEOUT_SECONDS" cargo run -- \
#     --root "$ROOT_DIR" \
#     --correction-json "$OUTPUT_DIR/correction_intelligence.json" \
#     --verification-policy-json "$OUTPUT_DIR/verification_policy.json" \
#     --report "$OUTPUT_DIR/executor_report.json" \
#     --verification-report "$OUTPUT_DIR/executor_verification_results.json" \
#     --rollback-log "$OUTPUT_DIR/executor_rollback_log.json" \
#     --diff-report "$OUTPUT_DIR/executor_diff_report.json" \
#     --input-report "$OUTPUT_DIR/executor_input_report.json" \
#     --print-diffs \
#     --diff-limit 5 \
#     --dry-run

# 2) Apply changes (mutations + verification)
# timeout "$TIMEOUT_SECONDS" cargo run -- \
#     --root "$ROOT_DIR" \
#     --correction-json "$OUTPUT_DIR/correction_intelligence.json" \
#     --verification-policy-json "$OUTPUT_DIR/verification_policy.json" \
#     --report "$OUTPUT_DIR/executor_report.json" \
#     --verification-report "$OUTPUT_DIR/executor_verification_results.json" \
#     --rollback-log "$OUTPUT_DIR/executor_rollback_log.json" \
#     --diff-report "$OUTPUT_DIR/executor_diff_report.json" \
#     --input-report "$OUTPUT_DIR/executor_input_report.json"
#     --print-diffs \
#     --diff-limit 5
#     --apply-path-coherence

# 3) Apply changes without verification
# timeout "$TIMEOUT_SECONDS" cargo run -- \
#     --root "$ROOT_DIR" \
#     --correction-json "$OUTPUT_DIR/correction_intelligence.json" \
#     --verification-policy-json "$OUTPUT_DIR/verification_policy.json" \
#     --report "$OUTPUT_DIR/executor_report.json" \
#     --verification-report "$OUTPUT_DIR/executor_verification_results.json" \
#     --rollback-log "$OUTPUT_DIR/executor_rollback_log.json" \
#     --diff-report "$OUTPUT_DIR/executor_diff_report.json" \
#     --input-report "$OUTPUT_DIR/executor_input_report.json" \
#     --print-diffs \
#     --diff-limit 5 \
#     --no-verify
#     --apply-path-coherence

# 4) MoveToLayer expansion (BundleMove allowlist, dry run)
OUTPUT_DIR="$ROOT_DIR/docs/97_correction_intelligence/slice_one_per_prefix"
timeout "$TIMEOUT_SECONDS" cargo run -- \
    --root "$ROOT_DIR" \
    --correction-json "$OUTPUT_DIR/correction_intelligence.json" \
    --verification-policy-json "$OUTPUT_DIR/verification_policy.json" \
    --report "$OUTPUT_DIR/executor_report.json" \
    --verification-report "$OUTPUT_DIR/executor_verification_results.json" \
    --rollback-log "$OUTPUT_DIR/executor_rollback_log.json" \
    --diff-report "$OUTPUT_DIR/executor_diff_report.json" \
    --input-report "$OUTPUT_DIR/executor_input_report.json" \
    --print-diffs \
    --diff-limit 5 \
    --apply-dependency-plan
