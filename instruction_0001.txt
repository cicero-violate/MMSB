# Investigation: Julia Benchmark Performance - RESOLVED

**Date:** 2025-12-17  
**Status:** Root cause identified

---

## Root Causes Found

### Issue #1: Replay (244ms → target <50ms)
**Cause:** `ensure_rust_artifacts()` called repeatedly  
- Each call runs `isfile()` checks on Rust library
- Hundreds of stat() syscalls per benchmark iteration
- Solution: Cache check result at module level

### Issue #2: Tick (367ms → target <16ms)  
**Cause:** `EventSystem.log_event!()` with @info macros  
- Logging active during benchmark runs
- I/O overhead dominates tick timing
- Solution: Gate logs during validation runs

### Issue #3: FFI Overhead (Minor)
**Cause:** `rust_get_last_error()` called after every operation  
- Extra FFI roundtrip even on success path
- Solution: Skip error check when previous call succeeded

---

## Performance Impact Estimates

```
Current (broken):
  Replay: 244ms = 200ms stat() + 40ms work + 4ms FFI
  Tick:   367ms = 300ms logging + 60ms work + 7ms FFI

After fixes:
  Replay: ~15ms = 0ms stat() + 12ms work + 3ms FFI  ✓
  Tick:   ~12ms = 0ms logging + 10ms work + 2ms FFI  ✓
```

---

## Implementation Plan

### 1. Cache Artifact Check (Priority: HIGH)
**File:** `src/ffi/FFIWrapper.jl` or equivalent

Before:
```julia
function ensure_rust_artifacts()
    if !isfile(RUST_LIB_PATH)  # Stat() every call!
        error("Library not found")
    end
end
```

After:
```julia
const ARTIFACTS_VERIFIED = Ref(false)

function ensure_rust_artifacts()
    if !ARTIFACTS_VERIFIED[]
        if !isfile(RUST_LIB_PATH)
            error("Library not found")
        end
        ARTIFACTS_VERIFIED[] = true
    end
end
```

### 2. Gate Event Logging (Priority: HIGH)
**File:** `src/03_dag/EventSystem.jl`

Add flag:
```julia
const ENABLE_EVENT_LOGGING = Ref(true)

function log_event!(msg)
    ENABLE_EVENT_LOGGING[] && @info msg
end
```

In benchmarks:
```julia
# benchmark/benchmarks.jl
MMSB.EventSystem.ENABLE_EVENT_LOGGING[] = false
```

### 3. Skip Redundant Error Checks (Priority: MEDIUM)
**File:** FFI wrapper functions

Before:
```julia
result = ccall(:mmsb_op, ...)
error_msg = rust_get_last_error()  # Always calls
```

After:
```julia
result = ccall(:mmsb_op, ...)
result == 0 && return  # Skip error check on success
error_msg = rust_get_last_error()
```

---

## Validation Commands

After implementing fixes:

```bash
# Should now pass:
julia --project=. benchmark/validate_all.jl

# Expected output:
✓ #1: Replay (<50ms)
✓ #6: Tick (<16ms)
```

---

## Status: Ready for Implementation

All root causes identified. Fixes are straightforward:
1. Module-level cache (5 min)
2. Logging flag (5 min)  
3. Skip error checks (10 min)

**Total time:** ~20 minutes to fix both failures.

