# Functions G-M

## Layer: 01_page

### Rust Functions

#### `generate_mask`

- **File:** MMSB/src/01_page/simd_mask.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `map`
  - `zip`
  - `iter`
  - `iter`

#### `load_checkpoint`

- **File:** MMSB/src/01_page/checkpoint.rs:0
- **Visibility:** Public
- **Calls:**
  - `BufReader::new`
  - `File::open`
  - `read_exact`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Vec::with_capacity`
  - `read_exact`
  - `PageID`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `i32::from_le_bytes`
  - `map_err`
  - `PageLocation::from_tag`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `push`
  - `restore_from_snapshot`
  - `Ok`
  - `Err`
  - `std::io::Error::new`

#### `merge_deltas`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Public
- **Calls:**
  - `merge`

#### `merge_dense_avx2`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Private
- **Calls:**
  - `min`
  - `len`
  - `len`
  - `_mm256_loadu_si256`
  - `add`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `add`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `as_ptr`
  - `_mm256_blendv_epi8`
  - `_mm256_storeu_si256`
  - `add`
  - `as_mut_ptr`
  - `_mm256_or_si256`
  - `_mm256_storeu_si256`
  - `as_mut_ptr`

#### `merge_dense_avx512`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Private
- **Calls:**
  - `min`
  - `len`
  - `len`
  - `_mm512_loadu_si512`
  - `add`
  - `as_ptr`
  - `_mm512_loadu_si512`
  - `add`
  - `as_ptr`
  - `_mm512_loadu_si512`
  - `as_ptr`
  - `_mm512_test_epi8_mask`
  - `_mm512_mask_blend_epi8`
  - `_mm512_storeu_si512`
  - `add`
  - `as_mut_ptr`

#### `merge_dense_simd`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Public
- **Calls:**
  - `merge_dense_avx512`
  - `merge_dense_avx2`
  - `min`
  - `len`
  - `len`

## Layer: 03_dag

### Rust Functions

#### `has_cycle`

- **File:** MMSB/src/03_dag/cycle_detection.rs:0
- **Visibility:** Public
- **Calls:**
  - `clone`
  - `read`
  - `HashMap::new`
  - `get`
  - `insert`
  - `get`
  - `dfs`
  - `insert`
  - `keys`
  - `dfs`

## Layer: root

### Rust Functions

#### `log_error_code`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `kind`

#### `main`

- **File:** MMSB/build.rs:0
- **Visibility:** Private

#### `mask_from_bytes`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `is_null`
  - `Vec::new`
  - `std::slice::from_raw_parts`
  - `saturating_mul`
  - `collect`
  - `map`
  - `iter`
  - `Vec::with_capacity`
  - `saturating_mul`
  - `push`
  - `len`
  - `truncate`

#### `mmsb_allocator_allocate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `PageHandle::null`
  - `set_last_error`
  - `PageHandle::null`
  - `PageLocation::from_tag`
  - `Some`
  - `allocate_raw`
  - `PageID`
  - `set_last_error`
  - `PageHandle::null`

#### `mmsb_allocator_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_allocator_get_page`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `PageHandle::null`
  - `acquire_page`
  - `PageID`
  - `set_last_error`
  - `PageHandle::null`

#### `mmsb_allocator_list_pages`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `page_infos`
  - `min`
  - `len`
  - `with`
  - `borrow_mut`
  - `clear`
  - `extend`
  - `map`
  - `take`
  - `iter`
  - `clone`
  - `enumerate`
  - `take`
  - `iter`
  - `is_empty`
  - `std::ptr::null`
  - `as_ptr`
  - `len`
  - `add`

#### `mmsb_allocator_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `PageAllocatorConfig::default`
  - `PageAllocator::new`
  - `Box::new`
  - `Box::into_raw`

#### `mmsb_allocator_page_count`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_allocator_release`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `release`
  - `PageID`

#### `mmsb_checkpoint_load`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `to_str`
  - `CStr::from_ptr`
  - `is_null`
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `checkpoint::load_checkpoint`
  - `set_last_error`

#### `mmsb_checkpoint_write`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `to_str`
  - `CStr::from_ptr`
  - `is_null`
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `checkpoint::write_checkpoint`
  - `set_last_error`

#### `mmsb_delta_apply`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `apply_delta`
  - `set_last_error`

#### `mmsb_delta_copy_intent_metadata`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `as_bytes`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_copy_mask`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `min`
  - `len`
  - `take`
  - `enumerate`
  - `iter`
  - `add`

#### `mmsb_delta_copy_payload`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_copy_source`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `as_bytes`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_epoch`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_delta_id`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_intent_metadata_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`

#### `mmsb_delta_is_sparse`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_mask_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `mask_from_bytes`
  - `vec_from_ptr`
  - `is_null`
  - `to_string`
  - `to_string`
  - `to_string_lossy`
  - `CStr::from_ptr`
  - `DeltaID`
  - `PageID`
  - `into`
  - `Source`
  - `Box::new`
  - `Box::into_raw`

#### `mmsb_delta_page_id`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_payload_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_set_intent_metadata`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `is_null`
  - `std::slice::from_raw_parts`
  - `std::str::from_utf8`
  - `Some`
  - `to_string`
  - `set_last_error`

#### `mmsb_delta_source_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_timestamp`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_error_is_fatal`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_fatal`

#### `mmsb_error_is_retryable`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_retryable`

#### `mmsb_get_last_error`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `with`
  - `borrow_mut`

#### `mmsb_page_epoch`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `epoch`

#### `mmsb_page_metadata_export`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `metadata_blob`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_page_metadata_import`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `std::slice::from_raw_parts`
  - `set_metadata_blob`
  - `set_last_error`

#### `mmsb_page_metadata_size`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`
  - `metadata_blob`

#### `mmsb_page_read`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `size`
  - `is_null`
  - `set_last_error`
  - `size`
  - `min`
  - `data_slice`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_page_write_masked`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `mask_from_bytes`
  - `vec_from_ptr`
  - `DeltaID`
  - `into`
  - `Source`
  - `into`
  - `apply_delta`
  - `set_last_error`

#### `mmsb_semiring_boolean_accumulate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `accumulate`

#### `mmsb_semiring_boolean_fold_add`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `zero`
  - `slice_from_ptr`
  - `map`
  - `iter`
  - `fold_add`

#### `mmsb_semiring_boolean_fold_mul`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `one`
  - `slice_from_ptr`
  - `map`
  - `iter`
  - `fold_mul`

#### `mmsb_semiring_tropical_accumulate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `accumulate`

#### `mmsb_semiring_tropical_fold_add`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `zero`
  - `slice_from_ptr`
  - `fold_add`
  - `copied`
  - `iter`

#### `mmsb_semiring_tropical_fold_mul`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `one`
  - `slice_from_ptr`
  - `fold_mul`
  - `copied`
  - `iter`

#### `mmsb_tlog_append`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `append`
  - `clone`
  - `set_last_error`
  - `log_error_code`

#### `mmsb_tlog_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_tlog_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `TLogHandle::null`
  - `CStr::from_ptr`
  - `to_str`
  - `set_last_error`
  - `TLogHandle::null`
  - `to_owned`
  - `TransactionLog::new`
  - `Box::into_raw`
  - `Box::new`
  - `set_last_error`
  - `log_error_code`
  - `TLogHandle::null`

#### `mmsb_tlog_reader_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_tlog_reader_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `TLogReaderHandle::null`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `TLogReaderHandle::null`
  - `TransactionLogReader::open`
  - `Box::into_raw`
  - `Box::new`
  - `set_last_error`
  - `log_error_code`
  - `TLogReaderHandle::null`

#### `mmsb_tlog_reader_next`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `DeltaHandle::null`
  - `next`
  - `Box::new`
  - `Box::into_raw`
  - `DeltaHandle::null`
  - `set_last_error`
  - `log_error_code`
  - `DeltaHandle::null`

#### `mmsb_tlog_summary`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `set_last_error`
  - `crate::page::tlog::summary`
  - `set_last_error`
  - `log_error_code`

