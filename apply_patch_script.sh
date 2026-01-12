#!/usr/bin/env bash
set -u

MODE="apply"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

PATCHES=(
  "txt01.patch"
  "txt02.patch"
  "txt03.patch"
  "txt04.patch"
  "txt05.patch"
  "txt06.patch"
  "txt07.patch"
  "txt08.patch"
  "txt09.patch"
  "txt10.patch"
)

echo "=== apply_patch execution start (mode: $MODE) ==="
echo

APPLIED=()
FAILED=()
SKIPPED=()

for patch in "${PATCHES[@]}"; do
  echo "-> Processing: $patch"

  if [[ ! -f "$patch" ]]; then
    echo "   [SKIPPED] file not found"
    SKIPPED+=("$patch (missing)")
    echo
    continue
  fi

  if [[ "$MODE" == "check" ]]; then
    # Snapshot current state
    git update-index -q --refresh
    git checkout -f -- . >/dev/null 2>&1

    OUTPUT=$(apply_patch < "$patch" 2>&1)
    STATUS=$?

    # Roll back unconditionally
    git checkout -f -- . >/dev/null 2>&1

    if [[ $STATUS -eq 0 ]]; then
      echo "   [WOULD APPLY]"
      APPLIED+=("$patch (would apply)")
    else
      if echo "$OUTPUT" | grep -qiE "already applied|reversed|skipping patch"; then
        echo "   [ALREADY APPLIED]"
        SKIPPED+=("$patch (already applied)")
      else
        echo "   [WOULD FAIL]"
        echo "   ---- apply_patch output ----"
        echo "$OUTPUT"
        echo "   ----------------------------"
        FAILED+=("$patch")
      fi
    fi

  else
    OUTPUT=$(apply_patch < "$patch" 2>&1)
    STATUS=$?

    if [[ $STATUS -eq 0 ]]; then
      echo "   [APPLIED]"
      APPLIED+=("$patch")
    else
      if echo "$OUTPUT" | grep -qiE "already applied|reversed|skipping patch"; then
        echo "   [SKIPPED] already applied"
        SKIPPED+=("$patch (already applied)")
      else
        echo "   [FAILED]"
        echo "   ---- apply_patch output ----"
        echo "$OUTPUT"
        echo "   ----------------------------"
        FAILED+=("$patch")
      fi
    fi
  fi

  echo
done

echo "=== execution summary (mode: $MODE) ==="
echo "Applied : ${#APPLIED[@]}"
for p in "${APPLIED[@]}"; do echo "  + $p"; done

echo
echo "Skipped : ${#SKIPPED[@]}"
for p in "${SKIPPED[@]}"; do echo "  ~ $p"; done

echo
echo "Failed  : ${#FAILED[@]}"
for p in "${FAILED[@]}"; do echo "  - $p"; done

echo
if [[ ${#FAILED[@]} -ne 0 ]]; then
  echo "Result: FAILURE"
  exit 1
else
  echo "Result: SUCCESS"
  exit 0
fi
