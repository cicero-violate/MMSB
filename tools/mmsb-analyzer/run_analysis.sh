#!/bin/bash
# Run MMSB analyzer on this project
# Generic version - works for both mmsb-analyzer and mmsb-executor

set -euo pipefail

# Auto-detect root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Auto-detect analyzer directory (works from any MMSB tool)
ANALYZER_DIR="$(cd "$SCRIPT_DIR/../mmsb-analyzer" && pwd)"

# Output directory
OUTPUT_DIR="$ROOT_DIR/docs"
CORRECTION_INTELLIGENCE="$OUTPUT_DIR/97_correction_intelligence/correction_intelligence.json"
ADMISSION_ARTIFACT="$ROOT_DIR/admission_composition.json"

cd "$ROOT_DIR"

# Pick one (uncomment the block you want):
# 1) Full workflow: build + analyze + TODO report with severity exit codes
cargo run --manifest-path "$ANALYZER_DIR/xtask/Cargo.toml" -- \
    check \
    --root "$ROOT_DIR" \
    --output "$OUTPUT_DIR" \
    --skip-julia \
    --dead-code \
    --dead-code-filter \
    --dead-code-json "$OUTPUT_DIR/98_dead_code/dead_code_full.json" \
    --dead-code-summary "$OUTPUT_DIR/98_dead_code/dead_code_summary.md" \
    --dead-code-summary-limit 50 \
    --dead-code-policy "$OUTPUT_DIR/98_dead_code/dead_code_policy.txt" \
    --correction-intelligence \
    --correction-json "$OUTPUT_DIR/97_correction_intelligence/correction_intelligence.json" \
    --verification-policy-json "$OUTPUT_DIR/97_correction_intelligence/verification_policy.json" \
    --correction-path-slice \
    --correction-visibility-slice \
    "$@" | tee "$ROOT_DIR/report_check.txt"

# ============================================================================
# PHASE 6.5 BATCH ADMISSION (CATEGORY 1 WIRING)
# ============================================================================
# Run batch-level admission after correction intelligence is generated.
# Always emit admission_composition.json.
# Do not interpret results. Do not suppress failures. Do not auto-execute.
# ============================================================================

echo ""
echo "============================================================================"
echo "Running PHASE 6.5 Batch Admission"
echo "============================================================================"

# Build the admission runner if needed
cargo build --example run_batch_admission --manifest-path "$ANALYZER_DIR/Cargo.toml"

# Run batch admission (unconditionally)
if [ -f "$CORRECTION_INTELLIGENCE" ]; then
    "$ANALYZER_DIR/target/debug/examples/run_batch_admission" \
        "$CORRECTION_INTELLIGENCE" \
        "$ADMISSION_ARTIFACT"

    echo "============================================================================"
    echo "Admission artifact: $ADMISSION_ARTIFACT"
    echo "============================================================================"
else
    echo "⚠️  Warning: Correction intelligence not found at $CORRECTION_INTELLIGENCE"
    echo "   Skipping batch admission"
fi

# 2) Analyze only (regenerate docs, no TODO report)
# cargo run --manifest-path "$ANALYZER_DIR/xtask/Cargo.toml" -- \
#     analyze \
#     --root "$ROOT_DIR" \
#     --output "$OUTPUT_DIR" \
#     --skip-julia \
#     --dead-code \
#     --dead-code-filter \
#     --dead-code-json "$OUTPUT_DIR/98_dead_code/dead_code_full.json" \
#     --dead-code-summary "$OUTPUT_DIR/98_dead_code/dead_code_summary.md" \
#     --dead-code-summary-limit 50 \
#     --dead-code-policy "$OUTPUT_DIR/98_dead_code/dead_code_policy.txt" \
#     --correction-intelligence \
#     --correction-json "$OUTPUT_DIR/97_correction_intelligence/correction_intelligence.json" \
#     --verification-policy-json "$OUTPUT_DIR/97_correction_intelligence/verification_policy.json" \
#     --correction-path-slice \
#     --correction-visibility-slice \
#     "$@" | tee "$ROOT_DIR/report_analyze.txt"

# 3) Report only (parse existing docs, no re-analysis)
# cargo run --manifest-path "$ANALYZER_DIR/xtask/Cargo.toml" -- \
#     report \
#     --docs-dir "$OUTPUT_DIR" \
#     "$@" | tee "$ROOT_DIR/report_only.txt"
