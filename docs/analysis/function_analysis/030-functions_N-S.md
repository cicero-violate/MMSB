# Functions N-S

## Layer: 01_page

### Rust Functions

#### `now_ns`

- **File:** MMSB/src/01_page/delta.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_nanos`
  - `unwrap_or_default`
  - `duration_since`
  - `SystemTime::now`

#### `read_bytes`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `to_vec`
  - `Ok`

#### `read_frame`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Private
- **Calls:**
  - `read_exact`
  - `kind`
  - `Ok`
  - `Err`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `collect`
  - `map`
  - `iter`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Source`
  - `to_string`
  - `String::from_utf8_lossy`
  - `is_err`
  - `read_exact`
  - `Ok`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Some`
  - `to_string`
  - `String::from_utf8_lossy`
  - `Ok`
  - `Some`
  - `DeltaID`
  - `u64::from_le_bytes`
  - `PageID`
  - `u64::from_le_bytes`
  - `Epoch`
  - `u32::from_le_bytes`
  - `u64::from_le_bytes`

#### `read_log`

- **File:** MMSB/src/01_page/tlog_serialization.rs:0
- **Visibility:** Public
- **Calls:**
  - `File::open`
  - `BufReader::new`
  - `read_exact`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `Vec::new`
  - `is_err`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `collect`
  - `map`
  - `iter`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Source`
  - `to_string`
  - `String::from_utf8_lossy`
  - `is_err`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Some`
  - `to_string`
  - `String::from_utf8_lossy`
  - `push`
  - `DeltaID`
  - `u64::from_le_bytes`
  - `PageID`
  - `u64::from_le_bytes`
  - `Epoch`
  - `u32::from_le_bytes`
  - `Ok`

#### `read_u32`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `map_err`
  - `try_into`
  - `PageError::MetadataDecode`
  - `Ok`
  - `u32::from_le_bytes`

#### `serialize_frame`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Private
- **Calls:**
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `len`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `write_all`
  - `write_all`
  - `to_le_bytes`
  - `as_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`
  - `as_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `as_bytes`
  - `Ok`

#### `summary`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Public
- **Calls:**
  - `File::open`
  - `as_ref`
  - `kind`
  - `Err`
  - `Err`
  - `len`
  - `metadata`
  - `Ok`
  - `LogSummary::default`
  - `BufReader::new`
  - `validate_header`
  - `LogSummary::default`
  - `read_frame`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`
  - `as_bytes`
  - `len`
  - `len`
  - `max`
  - `Ok`

## Layer: 04_propagation

### Rust Functions

#### `passthrough`

- **File:** MMSB/src/04_propagation/propagation_fastpath.rs:0
- **Visibility:** Public

## Layer: root

### Rust Functions

#### `read_page`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `to_vec`
  - `data_slice`

#### `rejects_mismatched_dense_lengths`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `dense_delta`

#### `set_last_error`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `with`
  - `borrow_mut`

#### `slice_from_ptr`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Generics:** 'a, T
- **Calls:**
  - `is_null`
  - `slice::from_raw_parts`

