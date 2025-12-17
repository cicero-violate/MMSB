# Functions A-F

## Layer: 01_page

### Rust Functions

#### `allocate_zeroed`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `map_err`
  - `std::alloc::Layout::array`
  - `PageError::AllocError`
  - `std::alloc::alloc_zeroed`
  - `is_null`
  - `Err`
  - `PageError::AllocError`
  - `Ok`

#### `apply_log`

- **File:** MMSB/src/01_page/tlog_replay.rs:0
- **Visibility:** Public
- **Calls:**
  - `find`
  - `iter_mut`
  - `apply_delta`

#### `bitpack_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `enumerate`
  - `iter`

#### `bitunpack_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `enumerate`
  - `iter_mut`
  - `len`

#### `compact`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `to_vec`
  - `Vec::with_capacity`
  - `len`
  - `iter`
  - `next`
  - `push`
  - `clone`
  - `last_mut`
  - `merge`
  - `push`
  - `clone`

#### `compress_delta_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `collect`
  - `map`
  - `iter`
  - `encode_rle`
  - `bitpack_mask`
  - `len`
  - `max`
  - `len`

#### `decode_rle`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`

#### `encode_rle`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `is_empty`
  - `push`
  - `push`

## Layer: 02_semiring

### Rust Functions

#### `accumulate`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `add`
  - `mul`

#### `fold_add`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `fold`
  - `into_iter`
  - `zero`
  - `add`

#### `fold_mul`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `fold`
  - `into_iter`
  - `one`
  - `mul`

## Layer: 03_dag

### Rust Functions

#### `dfs`

- **File:** MMSB/src/03_dag/cycle_detection.rs:0
- **Visibility:** Private
- **Calls:**
  - `get`
  - `insert`
  - `get`
  - `dfs`
  - `insert`

## Layer: 04_propagation

### Rust Functions

#### `enqueue_sparse`

- **File:** MMSB/src/04_propagation/sparse_message_passing.rs:0
- **Visibility:** Public
- **Calls:**
  - `push`

## Layer: 06_utility

### Rust Functions

#### `cpu_has_avx2`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

#### `cpu_has_avx512`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

#### `cpu_has_sse42`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

## Layer: root

### Rust Functions

#### `convert_location`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageLocation::from_tag`

#### `dense_delta`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

#### `example_checkpoint`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private

#### `example_delta_operations`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private
- **Calls:**
  - `Delta::new_dense`
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

#### `example_page_allocation`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocatorConfig::default`
  - `PageAllocator::new`
  - `PageID`
  - `allocate_raw`
  - `Some`
  - `free`

