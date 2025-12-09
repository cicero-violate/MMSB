# MMSB Segfault Diagnostics
_Compilation of everything needed to capture, interpret, and act on the current crash._

## 1. Problem Statement
- **Symptom**: `julia --project=. test/runtests.jl` crashes with `signal 11` while executing `mmsb_page_read` inside `libmmsb_core.so`.
- **Confirmed Not the Cause**: Julia GC issues (GC.@preserve blocks added), NULL handles, invalid page states — all guarded.
- **Hypothesis Domain**: Rust Page/allocator internals (invalid pointer, corrupted struct, data slice misuse).

## 2. Required Commands
| Purpose                            | Command                                                        | Notes                           |                            |
| ---                                | ---                                                            | ---                             |                            |
| Run full diagnostic suite          | `julia --project=. test/runtests.jl 2>&1                       | tee diagnostic_output.log`      | Captures Julia + Rust logs |
| Rebuild Rust library after changes | `cargo build --release && cp target/release/libmmsb_core.so .` | Required before rerunning tests |                            |
| Optional GC stress (post-fix)      | `julia --project=. test/gc_stress_test.jl`                     | Validates GC safety             |                            |
| Optional sanitizer run             | `RUSTFLAGS="-Z sanitizer=address" cargo +nightly test`         | For allocator/FFI issues        |                            |

## 3. Interpreting Rust Logs
Rust instrumentation (see `src/ffi.rs`) prints structured sections:

```
=== mmsb_allocator_alloc START ===
  handle.ptr = 0x...
  page_id = ...
  ...
=== mmsb_allocator_alloc END (success) ===

=== mmsb_page_new START ===
...

=== mmsb_page_read START ===
  handle.ptr = 0x...
  dst = 0x...
  len = ...
  NULL check passed
  Dereferencing page handle...
  Page deref OK
  Page ID = ...
  Calling page.data_slice()...
  data_slice() returned successfully
  data.len() = ...
  Will copy N bytes
  Performing copy_nonoverlapping...
  Copy completed successfully
=== mmsb_page_read END (success, N bytes) ===
```

**Use the last emitted line before the crash to pinpoint the failure mode:**

| Last Log Line | Meaning | Likely Fix Area |
| --- | --- | --- |
| `Dereferencing page handle...` (no "OK") | Handle points to freed/corrupt Page | Check allocator returns + lifetime |
| `Calling page.data_slice()...` (no return) | Page struct corrupted | Inspect Page::new / metadata writes |
| `data_slice() returned successfully` (next line missing) | Vec contents corrupt | Investigate data buffer allocation |
| `Performing copy_nonoverlapping...` | Destination buffer invalid/alignment issue | Verify Julia buffer + sizes |

Record the observed failure mode + timestamp in the Findings section below.

## 4. Current Findings
- _Pending run — no diagnostic log captured yet._  
  - Owner: _Unassigned_  
  - Action: Execute Required Command #1 and update this section.

## 5. Completed Instrumentation
- **Julia**: `rust_page_read!` validated + `GC.@preserve` wrapping; Page state machine ensures only initialized pages are read.
- **Rust**:
  - `mmsb_page_read`, `mmsb_page_new`, `mmsb_allocator_alloc` log entry/exit + key checkpoints.
  - NULL-pointer and state validation before each unsafe dereference.
  - Allocator logs pointer addresses and page metadata immediately after allocation.

## 6. Next-Step Playbooks
1. **If invalid pointer**: 
   - Add targeted unit test for allocator path.
   - Inspect `allocate_raw`, ensure pages stay alive, verify `Box::into_raw` usage.
2. **If corrupted struct**:
   - Verify metadata writes / FFI imports aren't overwriting `Page`.
   - Consider adding guard fields (canaries) in Rust struct for debug builds.
3. **If copy step fails**:
   - Double-check Julia buffer size vs. Rust `bytes` count.
   - Confirm `Vector{UInt8}` length equals page size.
4. **If no crash**:
   - Rerun to confirm stability, then move to T8–T10 hardening.

## 7. Logging & Reporting
- Store raw logs at `pending_work/test_output.log` or `pending_work/diagnostic_output.log`.
- Summarize key findings (one paragraph) in this file with date/owner.
- Mirror task status in `DAG_PRIORITY.md`.

