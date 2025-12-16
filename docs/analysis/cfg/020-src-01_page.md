# CFG Group: src/01_page

## Function: `allocate_zeroed`

- File: MMSB/src/01_page/page.rs
- Branches: 1
- Loops: 0
- Nodes: 8
- Edges: 8

```mermaid
flowchart TD
    allocate_zeroed_0["ENTRY"]
    allocate_zeroed_1["let layout = std :: alloc :: Layout :: array :: < u8 > (size) . map_err (| _ | PageError :..."]
    allocate_zeroed_2["let ptr = unsafe { std :: alloc :: alloc_zeroed (layout) }"]
    allocate_zeroed_3["if ptr . is_null ()"]
    allocate_zeroed_4["return Err (PageError :: AllocError (err_code))"]
    allocate_zeroed_5["if join"]
    allocate_zeroed_6["Ok (ptr)"]
    allocate_zeroed_7["EXIT"]
    allocate_zeroed_0 --> allocate_zeroed_1
    allocate_zeroed_1 --> allocate_zeroed_2
    allocate_zeroed_2 --> allocate_zeroed_3
    allocate_zeroed_3 --> allocate_zeroed_4
    allocate_zeroed_4 --> allocate_zeroed_5
    allocate_zeroed_3 --> allocate_zeroed_5
    allocate_zeroed_5 --> allocate_zeroed_6
    allocate_zeroed_6 --> allocate_zeroed_7
```

## Function: `apply_log`

- File: MMSB/src/01_page/tlog_replay.rs
- Branches: 1
- Loops: 1
- Nodes: 7
- Edges: 8

```mermaid
flowchart TD
    apply_log_0["ENTRY"]
    apply_log_1["for delta in deltas"]
    apply_log_2["if let Some (page) = pages . iter_mut () . find (| p | p . id == delta . page..."]
    apply_log_3["let _ = page . apply_delta (delta)"]
    apply_log_4["if join"]
    apply_log_5["after for"]
    apply_log_6["EXIT"]
    apply_log_0 --> apply_log_1
    apply_log_1 --> apply_log_2
    apply_log_2 --> apply_log_3
    apply_log_3 --> apply_log_4
    apply_log_2 --> apply_log_4
    apply_log_4 --> apply_log_1
    apply_log_1 --> apply_log_5
    apply_log_5 --> apply_log_6
```

## Function: `bitpack_mask`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 1
- Loops: 1
- Nodes: 10
- Edges: 11

```mermaid
flowchart TD
    bitpack_mask_0["ENTRY"]
    bitpack_mask_1["let num_bytes = (mask . len () + 7) / 8"]
    bitpack_mask_2["let mut packed = vec ! [0u8 ; num_bytes]"]
    bitpack_mask_3["for (i , & bit) in mask . iter () . enumerate ()"]
    bitpack_mask_4["if bit"]
    bitpack_mask_5["packed [i / 8] |= 1 << (i % 8)"]
    bitpack_mask_6["if join"]
    bitpack_mask_7["after for"]
    bitpack_mask_8["packed"]
    bitpack_mask_9["EXIT"]
    bitpack_mask_0 --> bitpack_mask_1
    bitpack_mask_1 --> bitpack_mask_2
    bitpack_mask_2 --> bitpack_mask_3
    bitpack_mask_3 --> bitpack_mask_4
    bitpack_mask_4 --> bitpack_mask_5
    bitpack_mask_5 --> bitpack_mask_6
    bitpack_mask_4 --> bitpack_mask_6
    bitpack_mask_6 --> bitpack_mask_3
    bitpack_mask_3 --> bitpack_mask_7
    bitpack_mask_7 --> bitpack_mask_8
    bitpack_mask_8 --> bitpack_mask_9
```

## Function: `bitunpack_mask`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 1
- Loops: 1
- Nodes: 9
- Edges: 10

```mermaid
flowchart TD
    bitunpack_mask_0["ENTRY"]
    bitunpack_mask_1["for (i , out) in output . iter_mut () . enumerate ()"]
    bitunpack_mask_2["let byte_idx = i / 8"]
    bitunpack_mask_3["let bit_idx = i % 8"]
    bitunpack_mask_4["if byte_idx < packed . len ()"]
    bitunpack_mask_5["* out = (packed [byte_idx] & (1 << bit_idx)) != 0"]
    bitunpack_mask_6["if join"]
    bitunpack_mask_7["after for"]
    bitunpack_mask_8["EXIT"]
    bitunpack_mask_0 --> bitunpack_mask_1
    bitunpack_mask_1 --> bitunpack_mask_2
    bitunpack_mask_2 --> bitunpack_mask_3
    bitunpack_mask_3 --> bitunpack_mask_4
    bitunpack_mask_4 --> bitunpack_mask_5
    bitunpack_mask_5 --> bitunpack_mask_6
    bitunpack_mask_4 --> bitunpack_mask_6
    bitunpack_mask_6 --> bitunpack_mask_1
    bitunpack_mask_1 --> bitunpack_mask_7
    bitunpack_mask_7 --> bitunpack_mask_8
```

## Function: `compact`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 5
- Loops: 1
- Nodes: 22
- Edges: 27

```mermaid
flowchart TD
    compact_0["ENTRY"]
    compact_1["if deltas . len () <= 1"]
    compact_2["return deltas . to_vec ()"]
    compact_3["if join"]
    compact_4["let mut result = Vec :: with_capacity (deltas . len ())"]
    compact_5["let mut iter = deltas . iter ()"]
    compact_6["if let Some (first) = iter . next ()"]
    compact_7["result . push (first . clone ())"]
    compact_8["for delta in iter"]
    compact_9["if let Some (last) = result . last_mut ()"]
    compact_10["if last . page_id == delta . page_id"]
    compact_11["if let Ok (merged) = last . merge (delta)"]
    compact_12["* last = merged"]
    compact_13["continue"]
    compact_14["if join"]
    compact_15["if join"]
    compact_16["if join"]
    compact_17["result . push (delta . clone ())"]
    compact_18["after for"]
    compact_19["if join"]
    compact_20["result"]
    compact_21["EXIT"]
    compact_0 --> compact_1
    compact_1 --> compact_2
    compact_2 --> compact_3
    compact_1 --> compact_3
    compact_3 --> compact_4
    compact_4 --> compact_5
    compact_5 --> compact_6
    compact_6 --> compact_7
    compact_7 --> compact_8
    compact_8 --> compact_9
    compact_9 --> compact_10
    compact_10 --> compact_11
    compact_11 --> compact_12
    compact_12 --> compact_13
    compact_13 --> compact_14
    compact_11 --> compact_14
    compact_14 --> compact_15
    compact_10 --> compact_15
    compact_15 --> compact_16
    compact_9 --> compact_16
    compact_16 --> compact_17
    compact_17 --> compact_8
    compact_8 --> compact_18
    compact_18 --> compact_19
    compact_6 --> compact_19
    compact_19 --> compact_20
    compact_20 --> compact_21
```

## Function: `compress_delta_mask`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    compress_delta_mask_0["ENTRY"]
    compress_delta_mask_1["let original = mask . len ()"]
    compress_delta_mask_2["let compressed = match mode { CompressionMode :: None => mask . iter () . map (| & b | b as u8..."]
    compress_delta_mask_3["let stats = CompressionStats { original_size : original , compressed_size : compressed . ..."]
    compress_delta_mask_4["(compressed , stats)"]
    compress_delta_mask_5["EXIT"]
    compress_delta_mask_0 --> compress_delta_mask_1
    compress_delta_mask_1 --> compress_delta_mask_2
    compress_delta_mask_2 --> compress_delta_mask_3
    compress_delta_mask_3 --> compress_delta_mask_4
    compress_delta_mask_4 --> compress_delta_mask_5
```

## Function: `decode_rle`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 1
- Loops: 2
- Nodes: 14
- Edges: 16

```mermaid
flowchart TD
    decode_rle_0["ENTRY"]
    decode_rle_1["let mut pos = 0"]
    decode_rle_2["for & byte in encoded"]
    decode_rle_3["let is_zero = (byte & 0x80) != 0"]
    decode_rle_4["let count = (byte & 0x7F) as usize"]
    decode_rle_5["let value = ! is_zero"]
    decode_rle_6["for _ in 0 .. count"]
    decode_rle_7["if pos < output . len ()"]
    decode_rle_8["output [pos] = value"]
    decode_rle_9["pos += 1"]
    decode_rle_10["if join"]
    decode_rle_11["after for"]
    decode_rle_12["after for"]
    decode_rle_13["EXIT"]
    decode_rle_0 --> decode_rle_1
    decode_rle_1 --> decode_rle_2
    decode_rle_2 --> decode_rle_3
    decode_rle_3 --> decode_rle_4
    decode_rle_4 --> decode_rle_5
    decode_rle_5 --> decode_rle_6
    decode_rle_6 --> decode_rle_7
    decode_rle_7 --> decode_rle_8
    decode_rle_8 --> decode_rle_9
    decode_rle_9 --> decode_rle_10
    decode_rle_7 --> decode_rle_10
    decode_rle_10 --> decode_rle_6
    decode_rle_6 --> decode_rle_11
    decode_rle_11 --> decode_rle_2
    decode_rle_2 --> decode_rle_12
    decode_rle_12 --> decode_rle_13
```

## Function: `encode_rle`

- File: MMSB/src/01_page/tlog_compression.rs
- Branches: 2
- Loops: 1
- Nodes: 18
- Edges: 20

```mermaid
flowchart TD
    encode_rle_0["ENTRY"]
    encode_rle_1["let mut encoded = Vec :: new ()"]
    encode_rle_2["if mask . is_empty ()"]
    encode_rle_3["return encoded"]
    encode_rle_4["if join"]
    encode_rle_5["let mut current = mask [0]"]
    encode_rle_6["let mut count : u8 = 1"]
    encode_rle_7["for & bit in & mask [1 ..]"]
    encode_rle_8["if bit == current && count < 255"]
    encode_rle_9["count += 1"]
    encode_rle_10["encoded . push (if current { count } else { count | 0x80 })"]
    encode_rle_11["current = bit"]
    encode_rle_12["count = 1"]
    encode_rle_13["if join"]
    encode_rle_14["after for"]
    encode_rle_15["encoded . push (if current { count } else { count | 0x80 })"]
    encode_rle_16["encoded"]
    encode_rle_17["EXIT"]
    encode_rle_0 --> encode_rle_1
    encode_rle_1 --> encode_rle_2
    encode_rle_2 --> encode_rle_3
    encode_rle_3 --> encode_rle_4
    encode_rle_2 --> encode_rle_4
    encode_rle_4 --> encode_rle_5
    encode_rle_5 --> encode_rle_6
    encode_rle_6 --> encode_rle_7
    encode_rle_7 --> encode_rle_8
    encode_rle_8 --> encode_rle_9
    encode_rle_8 --> encode_rle_10
    encode_rle_10 --> encode_rle_11
    encode_rle_11 --> encode_rle_12
    encode_rle_9 --> encode_rle_13
    encode_rle_12 --> encode_rle_13
    encode_rle_13 --> encode_rle_7
    encode_rle_7 --> encode_rle_14
    encode_rle_14 --> encode_rle_15
    encode_rle_15 --> encode_rle_16
    encode_rle_16 --> encode_rle_17
```

## Function: `generate_mask`

- File: MMSB/src/01_page/simd_mask.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    generate_mask_0["ENTRY"]
    generate_mask_1["macro assert_eq"]
    generate_mask_2["old . iter () . zip (new . iter ()) . map (| (a , b) | a != b) . collect ()"]
    generate_mask_3["EXIT"]
    generate_mask_0 --> generate_mask_1
    generate_mask_1 --> generate_mask_2
    generate_mask_2 --> generate_mask_3
```

## Function: `load_checkpoint`

- File: MMSB/src/01_page/checkpoint.rs
- Branches: 4
- Loops: 1
- Nodes: 62
- Edges: 65

```mermaid
flowchart TD
    load_checkpoint_0["ENTRY"]
    load_checkpoint_1["macro eprintln"]
    load_checkpoint_2["let mut reader = BufReader :: new (File :: open (path) ?)"]
    load_checkpoint_3["let mut magic = [0u8 ; 8]"]
    load_checkpoint_4["reader . read_exact (& mut magic) ?"]
    load_checkpoint_5["if & magic != SNAPSHOT_MAGIC"]
    load_checkpoint_6["return Err (std :: io :: Error :: new (std :: io :: ErrorKind :: InvalidData , forma..."]
    load_checkpoint_7["if join"]
    load_checkpoint_8["macro eprintln"]
    load_checkpoint_9["let mut version_bytes = [0u8 ; 4]"]
    load_checkpoint_10["reader . read_exact (& mut version_bytes) ?"]
    load_checkpoint_11["let version = u32 :: from_le_bytes (version_bytes)"]
    load_checkpoint_12["if version != SNAPSHOT_VERSION"]
    load_checkpoint_13["return Err (std :: io :: Error :: new (std :: io :: ErrorKind :: InvalidData , forma..."]
    load_checkpoint_14["if join"]
    load_checkpoint_15["macro eprintln"]
    load_checkpoint_16["let mut page_count_bytes = [0u8 ; 4]"]
    load_checkpoint_17["reader . read_exact (& mut page_count_bytes) ?"]
    load_checkpoint_18["let page_count = u32 :: from_le_bytes (page_count_bytes) as usize"]
    load_checkpoint_19["macro eprintln"]
    load_checkpoint_20["let mut log_offset_bytes = [0u8 ; 8]"]
    load_checkpoint_21["reader . read_exact (& mut log_offset_bytes) ?"]
    load_checkpoint_22["macro eprintln"]
    load_checkpoint_23["let mut snapshots = Vec :: with_capacity (page_count)"]
    load_checkpoint_24["for i in 0 .. page_count"]
    load_checkpoint_25["let mut id_bytes = [0u8 ; 8]"]
    load_checkpoint_26["reader . read_exact (& mut id_bytes) ?"]
    load_checkpoint_27["let id = PageID (u64 :: from_le_bytes (id_bytes))"]
    load_checkpoint_28["let mut size_bytes = [0u8 ; 8]"]
    load_checkpoint_29["reader . read_exact (& mut size_bytes) ?"]
    load_checkpoint_30["let size = u64 :: from_le_bytes (size_bytes) as usize"]
    load_checkpoint_31["let mut epoch_bytes = [0u8 ; 4]"]
    load_checkpoint_32["reader . read_exact (& mut epoch_bytes) ?"]
    load_checkpoint_33["let epoch = u32 :: from_le_bytes (epoch_bytes)"]
    load_checkpoint_34["let mut loc_bytes = [0u8 ; 4]"]
    load_checkpoint_35["reader . read_exact (& mut loc_bytes) ?"]
    load_checkpoint_36["let location_tag = i32 :: from_le_bytes (loc_bytes)"]
    load_checkpoint_37["let location = PageLocation :: from_tag (location_tag) . map_err (| _ | { std :: io :: Error..."]
    load_checkpoint_38["let mut metadata_len_bytes = [0u8 ; 4]"]
    load_checkpoint_39["reader . read_exact (& mut metadata_len_bytes) ?"]
    load_checkpoint_40["let metadata_len = u32 :: from_le_bytes (metadata_len_bytes) as usize"]
    load_checkpoint_41["let mut metadata_blob = vec ! [0u8 ; metadata_len]"]
    load_checkpoint_42["reader . read_exact (& mut metadata_blob) ?"]
    load_checkpoint_43["let mut data_len_bytes = [0u8 ; 4]"]
    load_checkpoint_44["reader . read_exact (& mut data_len_bytes) ?"]
    load_checkpoint_45["let data_len = u32 :: from_le_bytes (data_len_bytes) as usize"]
    load_checkpoint_46["let mut data = vec ! [0u8 ; data_len]"]
    load_checkpoint_47["reader . read_exact (& mut data) ?"]
    load_checkpoint_48["macro eprintln"]
    load_checkpoint_49["snapshots . push (PageSnapshotData { page_id : id , size , epoch , location ,..."]
    load_checkpoint_50["after for"]
    load_checkpoint_51["macro eprintln"]
    load_checkpoint_52["match allocator . restore_from_snapshot (snapshots)"]
    load_checkpoint_53["arm Ok (_)"]
    load_checkpoint_54["macro eprintln"]
    load_checkpoint_55["Ok (())"]
    load_checkpoint_56["arm Err (e)"]
    load_checkpoint_57["macro eprintln"]
    load_checkpoint_58["macro eprintln"]
    load_checkpoint_59["Err (std :: io :: Error :: new (std :: io :: ErrorKind :: Other , format ! ('..."]
    load_checkpoint_60["match join"]
    load_checkpoint_61["EXIT"]
    load_checkpoint_0 --> load_checkpoint_1
    load_checkpoint_1 --> load_checkpoint_2
    load_checkpoint_2 --> load_checkpoint_3
    load_checkpoint_3 --> load_checkpoint_4
    load_checkpoint_4 --> load_checkpoint_5
    load_checkpoint_5 --> load_checkpoint_6
    load_checkpoint_6 --> load_checkpoint_7
    load_checkpoint_5 --> load_checkpoint_7
    load_checkpoint_7 --> load_checkpoint_8
    load_checkpoint_8 --> load_checkpoint_9
    load_checkpoint_9 --> load_checkpoint_10
    load_checkpoint_10 --> load_checkpoint_11
    load_checkpoint_11 --> load_checkpoint_12
    load_checkpoint_12 --> load_checkpoint_13
    load_checkpoint_13 --> load_checkpoint_14
    load_checkpoint_12 --> load_checkpoint_14
    load_checkpoint_14 --> load_checkpoint_15
    load_checkpoint_15 --> load_checkpoint_16
    load_checkpoint_16 --> load_checkpoint_17
    load_checkpoint_17 --> load_checkpoint_18
    load_checkpoint_18 --> load_checkpoint_19
    load_checkpoint_19 --> load_checkpoint_20
    load_checkpoint_20 --> load_checkpoint_21
    load_checkpoint_21 --> load_checkpoint_22
    load_checkpoint_22 --> load_checkpoint_23
    load_checkpoint_23 --> load_checkpoint_24
    load_checkpoint_24 --> load_checkpoint_25
    load_checkpoint_25 --> load_checkpoint_26
    load_checkpoint_26 --> load_checkpoint_27
    load_checkpoint_27 --> load_checkpoint_28
    load_checkpoint_28 --> load_checkpoint_29
    load_checkpoint_29 --> load_checkpoint_30
    load_checkpoint_30 --> load_checkpoint_31
    load_checkpoint_31 --> load_checkpoint_32
    load_checkpoint_32 --> load_checkpoint_33
    load_checkpoint_33 --> load_checkpoint_34
    load_checkpoint_34 --> load_checkpoint_35
    load_checkpoint_35 --> load_checkpoint_36
    load_checkpoint_36 --> load_checkpoint_37
    load_checkpoint_37 --> load_checkpoint_38
    load_checkpoint_38 --> load_checkpoint_39
    load_checkpoint_39 --> load_checkpoint_40
    load_checkpoint_40 --> load_checkpoint_41
    load_checkpoint_41 --> load_checkpoint_42
    load_checkpoint_42 --> load_checkpoint_43
    load_checkpoint_43 --> load_checkpoint_44
    load_checkpoint_44 --> load_checkpoint_45
    load_checkpoint_45 --> load_checkpoint_46
    load_checkpoint_46 --> load_checkpoint_47
    load_checkpoint_47 --> load_checkpoint_48
    load_checkpoint_48 --> load_checkpoint_49
    load_checkpoint_49 --> load_checkpoint_24
    load_checkpoint_24 --> load_checkpoint_50
    load_checkpoint_50 --> load_checkpoint_51
    load_checkpoint_51 --> load_checkpoint_52
    load_checkpoint_52 --> load_checkpoint_53
    load_checkpoint_53 --> load_checkpoint_54
    load_checkpoint_54 --> load_checkpoint_55
    load_checkpoint_52 --> load_checkpoint_56
    load_checkpoint_56 --> load_checkpoint_57
    load_checkpoint_57 --> load_checkpoint_58
    load_checkpoint_58 --> load_checkpoint_59
    load_checkpoint_55 --> load_checkpoint_60
    load_checkpoint_59 --> load_checkpoint_60
    load_checkpoint_60 --> load_checkpoint_61
```

## Function: `merge_deltas`

- File: MMSB/src/01_page/delta_merge.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    merge_deltas_0["ENTRY"]
    merge_deltas_1["first . merge (second)"]
    merge_deltas_2["EXIT"]
    merge_deltas_0 --> merge_deltas_1
    merge_deltas_1 --> merge_deltas_2
```

## Function: `merge_dense_avx2`

- File: MMSB/src/01_page/delta_merge.rs
- Branches: 1
- Loops: 4
- Nodes: 34
- Edges: 38

```mermaid
flowchart TD
    merge_dense_avx2_0["ENTRY"]
    merge_dense_avx2_1["let len = data_a . len () . min (data_b . len ())"]
    merge_dense_avx2_2["let mut i = 0"]
    merge_dense_avx2_3["while i + 32 <= len"]
    merge_dense_avx2_4["let va = _mm256_loadu_si256 (data_a . as_ptr () . add (i) as * const __m256i)"]
    merge_dense_avx2_5["let vb = _mm256_loadu_si256 (data_b . as_ptr () . add (i) as * const __m256i)"]
    merge_dense_avx2_6["let mut mask_a_bytes = [0u8 ; 32]"]
    merge_dense_avx2_7["let mut mask_b_bytes = [0u8 ; 32]"]
    merge_dense_avx2_8["for j in 0 .. 32"]
    merge_dense_avx2_9["mask_a_bytes [j] = mask_a [i + j] as u8 * 0xFF"]
    merge_dense_avx2_10["mask_b_bytes [j] = mask_b [i + j] as u8 * 0xFF"]
    merge_dense_avx2_11["after for"]
    merge_dense_avx2_12["let ma = _mm256_loadu_si256 (mask_a_bytes . as_ptr () as * const __m256i)"]
    merge_dense_avx2_13["let mb = _mm256_loadu_si256 (mask_b_bytes . as_ptr () as * const __m256i)"]
    merge_dense_avx2_14["let result = _mm256_blendv_epi8 (va , vb , mb)"]
    merge_dense_avx2_15["_mm256_storeu_si256 (out_data . as_mut_ptr () . add (i) as * mut __m256i , re..."]
    merge_dense_avx2_16["let mask_result = _mm256_or_si256 (ma , mb)"]
    merge_dense_avx2_17["let mut mask_out_bytes = [0u8 ; 32]"]
    merge_dense_avx2_18["_mm256_storeu_si256 (mask_out_bytes . as_mut_ptr () as * mut __m256i , mask_r..."]
    merge_dense_avx2_19["for j in 0 .. 32"]
    merge_dense_avx2_20["out_mask [i + j] = mask_out_bytes [j] != 0"]
    merge_dense_avx2_21["after for"]
    merge_dense_avx2_22["i += 32"]
    merge_dense_avx2_23["after while"]
    merge_dense_avx2_24["while i < len"]
    merge_dense_avx2_25["if mask_b [i]"]
    merge_dense_avx2_26["out_data [i] = data_b [i]"]
    merge_dense_avx2_27["out_mask [i] = true"]
    merge_dense_avx2_28["out_data [i] = data_a [i]"]
    merge_dense_avx2_29["out_mask [i] = mask_a [i]"]
    merge_dense_avx2_30["if join"]
    merge_dense_avx2_31["i += 1"]
    merge_dense_avx2_32["after while"]
    merge_dense_avx2_33["EXIT"]
    merge_dense_avx2_0 --> merge_dense_avx2_1
    merge_dense_avx2_1 --> merge_dense_avx2_2
    merge_dense_avx2_2 --> merge_dense_avx2_3
    merge_dense_avx2_3 --> merge_dense_avx2_4
    merge_dense_avx2_4 --> merge_dense_avx2_5
    merge_dense_avx2_5 --> merge_dense_avx2_6
    merge_dense_avx2_6 --> merge_dense_avx2_7
    merge_dense_avx2_7 --> merge_dense_avx2_8
    merge_dense_avx2_8 --> merge_dense_avx2_9
    merge_dense_avx2_9 --> merge_dense_avx2_10
    merge_dense_avx2_10 --> merge_dense_avx2_8
    merge_dense_avx2_8 --> merge_dense_avx2_11
    merge_dense_avx2_11 --> merge_dense_avx2_12
    merge_dense_avx2_12 --> merge_dense_avx2_13
    merge_dense_avx2_13 --> merge_dense_avx2_14
    merge_dense_avx2_14 --> merge_dense_avx2_15
    merge_dense_avx2_15 --> merge_dense_avx2_16
    merge_dense_avx2_16 --> merge_dense_avx2_17
    merge_dense_avx2_17 --> merge_dense_avx2_18
    merge_dense_avx2_18 --> merge_dense_avx2_19
    merge_dense_avx2_19 --> merge_dense_avx2_20
    merge_dense_avx2_20 --> merge_dense_avx2_19
    merge_dense_avx2_19 --> merge_dense_avx2_21
    merge_dense_avx2_21 --> merge_dense_avx2_22
    merge_dense_avx2_22 --> merge_dense_avx2_3
    merge_dense_avx2_3 --> merge_dense_avx2_23
    merge_dense_avx2_23 --> merge_dense_avx2_24
    merge_dense_avx2_24 --> merge_dense_avx2_25
    merge_dense_avx2_25 --> merge_dense_avx2_26
    merge_dense_avx2_26 --> merge_dense_avx2_27
    merge_dense_avx2_25 --> merge_dense_avx2_28
    merge_dense_avx2_28 --> merge_dense_avx2_29
    merge_dense_avx2_27 --> merge_dense_avx2_30
    merge_dense_avx2_29 --> merge_dense_avx2_30
    merge_dense_avx2_30 --> merge_dense_avx2_31
    merge_dense_avx2_31 --> merge_dense_avx2_24
    merge_dense_avx2_24 --> merge_dense_avx2_32
    merge_dense_avx2_32 --> merge_dense_avx2_33
```

## Function: `merge_dense_avx512`

- File: MMSB/src/01_page/delta_merge.rs
- Branches: 1
- Loops: 4
- Nodes: 29
- Edges: 33

```mermaid
flowchart TD
    merge_dense_avx512_0["ENTRY"]
    merge_dense_avx512_1["let len = data_a . len () . min (data_b . len ())"]
    merge_dense_avx512_2["let mut i = 0"]
    merge_dense_avx512_3["while i + 64 <= len"]
    merge_dense_avx512_4["let va = _mm512_loadu_si512 (data_a . as_ptr () . add (i) as * const __m512i)"]
    merge_dense_avx512_5["let vb = _mm512_loadu_si512 (data_b . as_ptr () . add (i) as * const __m512i)"]
    merge_dense_avx512_6["let mut mask_b_bytes = [0u8 ; 64]"]
    merge_dense_avx512_7["for j in 0 .. 64"]
    merge_dense_avx512_8["mask_b_bytes [j] = mask_b [i + j] as u8 * 0xFF"]
    merge_dense_avx512_9["after for"]
    merge_dense_avx512_10["let mb = _mm512_loadu_si512 (mask_b_bytes . as_ptr () as * const __m512i)"]
    merge_dense_avx512_11["let blend_mask = _mm512_test_epi8_mask (mb , mb)"]
    merge_dense_avx512_12["let result = _mm512_mask_blend_epi8 (blend_mask , va , vb)"]
    merge_dense_avx512_13["_mm512_storeu_si512 (out_data . as_mut_ptr () . add (i) as * mut __m512i , re..."]
    merge_dense_avx512_14["for j in 0 .. 64"]
    merge_dense_avx512_15["out_mask [i + j] = mask_a [i + j] || mask_b [i + j]"]
    merge_dense_avx512_16["after for"]
    merge_dense_avx512_17["i += 64"]
    merge_dense_avx512_18["after while"]
    merge_dense_avx512_19["while i < len"]
    merge_dense_avx512_20["if mask_b [i]"]
    merge_dense_avx512_21["out_data [i] = data_b [i]"]
    merge_dense_avx512_22["out_mask [i] = true"]
    merge_dense_avx512_23["out_data [i] = data_a [i]"]
    merge_dense_avx512_24["out_mask [i] = mask_a [i]"]
    merge_dense_avx512_25["if join"]
    merge_dense_avx512_26["i += 1"]
    merge_dense_avx512_27["after while"]
    merge_dense_avx512_28["EXIT"]
    merge_dense_avx512_0 --> merge_dense_avx512_1
    merge_dense_avx512_1 --> merge_dense_avx512_2
    merge_dense_avx512_2 --> merge_dense_avx512_3
    merge_dense_avx512_3 --> merge_dense_avx512_4
    merge_dense_avx512_4 --> merge_dense_avx512_5
    merge_dense_avx512_5 --> merge_dense_avx512_6
    merge_dense_avx512_6 --> merge_dense_avx512_7
    merge_dense_avx512_7 --> merge_dense_avx512_8
    merge_dense_avx512_8 --> merge_dense_avx512_7
    merge_dense_avx512_7 --> merge_dense_avx512_9
    merge_dense_avx512_9 --> merge_dense_avx512_10
    merge_dense_avx512_10 --> merge_dense_avx512_11
    merge_dense_avx512_11 --> merge_dense_avx512_12
    merge_dense_avx512_12 --> merge_dense_avx512_13
    merge_dense_avx512_13 --> merge_dense_avx512_14
    merge_dense_avx512_14 --> merge_dense_avx512_15
    merge_dense_avx512_15 --> merge_dense_avx512_14
    merge_dense_avx512_14 --> merge_dense_avx512_16
    merge_dense_avx512_16 --> merge_dense_avx512_17
    merge_dense_avx512_17 --> merge_dense_avx512_3
    merge_dense_avx512_3 --> merge_dense_avx512_18
    merge_dense_avx512_18 --> merge_dense_avx512_19
    merge_dense_avx512_19 --> merge_dense_avx512_20
    merge_dense_avx512_20 --> merge_dense_avx512_21
    merge_dense_avx512_21 --> merge_dense_avx512_22
    merge_dense_avx512_20 --> merge_dense_avx512_23
    merge_dense_avx512_23 --> merge_dense_avx512_24
    merge_dense_avx512_22 --> merge_dense_avx512_25
    merge_dense_avx512_24 --> merge_dense_avx512_25
    merge_dense_avx512_25 --> merge_dense_avx512_26
    merge_dense_avx512_26 --> merge_dense_avx512_19
    merge_dense_avx512_19 --> merge_dense_avx512_27
    merge_dense_avx512_27 --> merge_dense_avx512_28
```

## Function: `merge_dense_simd`

- File: MMSB/src/01_page/delta_merge.rs
- Branches: 3
- Loops: 1
- Nodes: 19
- Edges: 22

```mermaid
flowchart TD
    merge_dense_simd_0["ENTRY"]
    merge_dense_simd_1["if is_x86_feature_detected ! ('avx512f')"]
    merge_dense_simd_2["unsafe { merge_dense_avx512 (data_a , mask_a , data_b , mask_b , out_data , o..."]
    merge_dense_simd_3["return"]
    merge_dense_simd_4["if join"]
    merge_dense_simd_5["if is_x86_feature_detected ! ('avx2')"]
    merge_dense_simd_6["unsafe { merge_dense_avx2 (data_a , mask_a , data_b , mask_b , out_data , out..."]
    merge_dense_simd_7["return"]
    merge_dense_simd_8["if join"]
    merge_dense_simd_9["let len = data_a . len () . min (data_b . len ())"]
    merge_dense_simd_10["for i in 0 .. len"]
    merge_dense_simd_11["if mask_b [i]"]
    merge_dense_simd_12["out_data [i] = data_b [i]"]
    merge_dense_simd_13["out_mask [i] = true"]
    merge_dense_simd_14["out_data [i] = data_a [i]"]
    merge_dense_simd_15["out_mask [i] = mask_a [i]"]
    merge_dense_simd_16["if join"]
    merge_dense_simd_17["after for"]
    merge_dense_simd_18["EXIT"]
    merge_dense_simd_0 --> merge_dense_simd_1
    merge_dense_simd_1 --> merge_dense_simd_2
    merge_dense_simd_2 --> merge_dense_simd_3
    merge_dense_simd_3 --> merge_dense_simd_4
    merge_dense_simd_1 --> merge_dense_simd_4
    merge_dense_simd_4 --> merge_dense_simd_5
    merge_dense_simd_5 --> merge_dense_simd_6
    merge_dense_simd_6 --> merge_dense_simd_7
    merge_dense_simd_7 --> merge_dense_simd_8
    merge_dense_simd_5 --> merge_dense_simd_8
    merge_dense_simd_8 --> merge_dense_simd_9
    merge_dense_simd_9 --> merge_dense_simd_10
    merge_dense_simd_10 --> merge_dense_simd_11
    merge_dense_simd_11 --> merge_dense_simd_12
    merge_dense_simd_12 --> merge_dense_simd_13
    merge_dense_simd_11 --> merge_dense_simd_14
    merge_dense_simd_14 --> merge_dense_simd_15
    merge_dense_simd_13 --> merge_dense_simd_16
    merge_dense_simd_15 --> merge_dense_simd_16
    merge_dense_simd_16 --> merge_dense_simd_10
    merge_dense_simd_10 --> merge_dense_simd_17
    merge_dense_simd_17 --> merge_dense_simd_18
```

## Function: `now_ns`

- File: MMSB/src/01_page/delta.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    now_ns_0["ENTRY"]
    now_ns_1["SystemTime :: now () . duration_since (UNIX_EPOCH) . unwrap_or_default () . a..."]
    now_ns_2["EXIT"]
    now_ns_0 --> now_ns_1
    now_ns_1 --> now_ns_2
```

## Function: `read_bytes`

- File: MMSB/src/01_page/page.rs
- Branches: 1
- Loops: 0
- Nodes: 8
- Edges: 8

```mermaid
flowchart TD
    read_bytes_0["ENTRY"]
    read_bytes_1["if * cursor + len > blob . len ()"]
    read_bytes_2["return Err (PageError :: MetadataDecode ('metadata truncated'))"]
    read_bytes_3["if join"]
    read_bytes_4["let bytes = blob [* cursor .. * cursor + len] . to_vec ()"]
    read_bytes_5["* cursor += len"]
    read_bytes_6["Ok (bytes)"]
    read_bytes_7["EXIT"]
    read_bytes_0 --> read_bytes_1
    read_bytes_1 --> read_bytes_2
    read_bytes_2 --> read_bytes_3
    read_bytes_1 --> read_bytes_3
    read_bytes_3 --> read_bytes_4
    read_bytes_4 --> read_bytes_5
    read_bytes_5 --> read_bytes_6
    read_bytes_6 --> read_bytes_7
```

## Function: `read_frame`

- File: MMSB/src/01_page/tlog.rs
- Branches: 3
- Loops: 0
- Nodes: 37
- Edges: 38

```mermaid
flowchart TD
    read_frame_0["ENTRY"]
    read_frame_1["let mut delta_id = [0u8 ; 8]"]
    read_frame_2["match reader . read_exact (& mut delta_id)"]
    read_frame_3["arm Ok (())"]
    read_frame_4["arm Err (err) if guard"]
    read_frame_5["return Ok (None)"]
    read_frame_6["arm Err (err)"]
    read_frame_7["return Err (err)"]
    read_frame_8["match join"]
    read_frame_9["let mut page_id = [0u8 ; 8]"]
    read_frame_10["reader . read_exact (& mut page_id) ?"]
    read_frame_11["let mut epoch = [0u8 ; 4]"]
    read_frame_12["reader . read_exact (& mut epoch) ?"]
    read_frame_13["let mut mask_len_bytes = [0u8 ; 4]"]
    read_frame_14["reader . read_exact (& mut mask_len_bytes) ?"]
    read_frame_15["let mask_len = u32 :: from_le_bytes (mask_len_bytes) as usize"]
    read_frame_16["let mut mask_raw = vec ! [0u8 ; mask_len]"]
    read_frame_17["reader . read_exact (& mut mask_raw) ?"]
    read_frame_18["let mask = mask_raw . iter () . map (| b | * b != 0) . collect :: < Vec < bool > > ()"]
    read_frame_19["let mut payload_len_bytes = [0u8 ; 4]"]
    read_frame_20["reader . read_exact (& mut payload_len_bytes) ?"]
    read_frame_21["let payload_len = u32 :: from_le_bytes (payload_len_bytes) as usize"]
    read_frame_22["let mut payload = vec ! [0u8 ; payload_len]"]
    read_frame_23["reader . read_exact (& mut payload) ?"]
    read_frame_24["let mut sparse_flag = [0u8 ; 1]"]
    read_frame_25["reader . read_exact (& mut sparse_flag) ?"]
    read_frame_26["let mut timestamp_bytes = [0u8 ; 8]"]
    read_frame_27["reader . read_exact (& mut timestamp_bytes) ?"]
    read_frame_28["let mut source_len_bytes = [0u8 ; 4]"]
    read_frame_29["reader . read_exact (& mut source_len_bytes) ?"]
    read_frame_30["let source_len = u32 :: from_le_bytes (source_len_bytes) as usize"]
    read_frame_31["let mut source_buf = vec ! [0u8 ; source_len]"]
    read_frame_32["reader . read_exact (& mut source_buf) ?"]
    read_frame_33["let source = Source (String :: from_utf8_lossy (& source_buf) . to_string ())"]
    read_frame_34["let intent_metadata = if version >= 2 { let mut metadata_len_bytes = [0u8 ; 4] ; if reader . read_e..."]
    read_frame_35["Ok (Some (Delta { delta_id : DeltaID (u64 :: from_le_bytes (delta_id)) , page..."]
    read_frame_36["EXIT"]
    read_frame_0 --> read_frame_1
    read_frame_1 --> read_frame_2
    read_frame_2 --> read_frame_3
    read_frame_2 --> read_frame_4
    read_frame_4 --> read_frame_5
    read_frame_2 --> read_frame_6
    read_frame_6 --> read_frame_7
    read_frame_3 --> read_frame_8
    read_frame_5 --> read_frame_8
    read_frame_7 --> read_frame_8
    read_frame_8 --> read_frame_9
    read_frame_9 --> read_frame_10
    read_frame_10 --> read_frame_11
    read_frame_11 --> read_frame_12
    read_frame_12 --> read_frame_13
    read_frame_13 --> read_frame_14
    read_frame_14 --> read_frame_15
    read_frame_15 --> read_frame_16
    read_frame_16 --> read_frame_17
    read_frame_17 --> read_frame_18
    read_frame_18 --> read_frame_19
    read_frame_19 --> read_frame_20
    read_frame_20 --> read_frame_21
    read_frame_21 --> read_frame_22
    read_frame_22 --> read_frame_23
    read_frame_23 --> read_frame_24
    read_frame_24 --> read_frame_25
    read_frame_25 --> read_frame_26
    read_frame_26 --> read_frame_27
    read_frame_27 --> read_frame_28
    read_frame_28 --> read_frame_29
    read_frame_29 --> read_frame_30
    read_frame_30 --> read_frame_31
    read_frame_31 --> read_frame_32
    read_frame_32 --> read_frame_33
    read_frame_33 --> read_frame_34
    read_frame_34 --> read_frame_35
    read_frame_35 --> read_frame_36
```

## Function: `read_log`

- File: MMSB/src/01_page/tlog_serialization.rs
- Branches: 2
- Loops: 1
- Nodes: 49
- Edges: 51

```mermaid
flowchart TD
    read_log_0["ENTRY"]
    read_log_1["let file = File :: open (path) ?"]
    read_log_2["let mut reader = BufReader :: new (file)"]
    read_log_3["let mut magic = [0u8 ; 8]"]
    read_log_4["reader . read_exact (& mut magic) ?"]
    read_log_5["if & magic != b'MMSBLOG1'"]
    read_log_6["return Err (std :: io :: Error :: new (std :: io :: ErrorKind :: InvalidData , 'inva..."]
    read_log_7["if join"]
    read_log_8["let mut version_bytes = [0u8 ; 4]"]
    read_log_9["reader . read_exact (& mut version_bytes) ?"]
    read_log_10["let version = u32 :: from_le_bytes (version_bytes)"]
    read_log_11["let mut deltas = Vec :: new ()"]
    read_log_12["loop"]
    read_log_13["let mut delta_id = [0u8 ; 8]"]
    read_log_14["if reader . read_exact (& mut delta_id) . is_err ()"]
    read_log_15["break"]
    read_log_16["if join"]
    read_log_17["let mut page_id = [0u8 ; 8]"]
    read_log_18["reader . read_exact (& mut page_id) ?"]
    read_log_19["let mut epoch = [0u8 ; 4]"]
    read_log_20["reader . read_exact (& mut epoch) ?"]
    read_log_21["let mut mask_len_bytes = [0u8 ; 4]"]
    read_log_22["reader . read_exact (& mut mask_len_bytes) ?"]
    read_log_23["let mask_len = u32 :: from_le_bytes (mask_len_bytes) as usize"]
    read_log_24["let mut mask_bytes = vec ! [0u8 ; mask_len]"]
    read_log_25["reader . read_exact (& mut mask_bytes) ?"]
    read_log_26["let mask = mask_bytes . iter () . map (| b | * b != 0) . collect ()"]
    read_log_27["let mut payload_len_bytes = [0u8 ; 4]"]
    read_log_28["reader . read_exact (& mut payload_len_bytes) ?"]
    read_log_29["let payload_len = u32 :: from_le_bytes (payload_len_bytes) as usize"]
    read_log_30["let mut payload = vec ! [0u8 ; payload_len]"]
    read_log_31["reader . read_exact (& mut payload) ?"]
    read_log_32["let mut sparse_flag = [0u8 ; 1]"]
    read_log_33["reader . read_exact (& mut sparse_flag) ?"]
    read_log_34["let is_sparse = sparse_flag [0] != 0"]
    read_log_35["let mut timestamp_bytes = [0u8 ; 8]"]
    read_log_36["reader . read_exact (& mut timestamp_bytes) ?"]
    read_log_37["let timestamp = u64 :: from_le_bytes (timestamp_bytes)"]
    read_log_38["let mut source_len_bytes = [0u8 ; 4]"]
    read_log_39["reader . read_exact (& mut source_len_bytes) ?"]
    read_log_40["let source_len = u32 :: from_le_bytes (source_len_bytes) as usize"]
    read_log_41["let mut source_buf = vec ! [0u8 ; source_len]"]
    read_log_42["reader . read_exact (& mut source_buf) ?"]
    read_log_43["let source = Source (String :: from_utf8_lossy (& source_buf) . to_string ())"]
    read_log_44["let intent_metadata = if version >= 2 { let mut metadata_len_bytes = [0u8 ; 4] ; if reader . read_e..."]
    read_log_45["deltas . push (Delta { delta_id : DeltaID (u64 :: from_le_bytes (delta_id)) ,..."]
    read_log_46["loop break"]
    read_log_47["Ok (deltas)"]
    read_log_48["EXIT"]
    read_log_0 --> read_log_1
    read_log_1 --> read_log_2
    read_log_2 --> read_log_3
    read_log_3 --> read_log_4
    read_log_4 --> read_log_5
    read_log_5 --> read_log_6
    read_log_6 --> read_log_7
    read_log_5 --> read_log_7
    read_log_7 --> read_log_8
    read_log_8 --> read_log_9
    read_log_9 --> read_log_10
    read_log_10 --> read_log_11
    read_log_11 --> read_log_12
    read_log_12 --> read_log_13
    read_log_13 --> read_log_14
    read_log_14 --> read_log_15
    read_log_15 --> read_log_16
    read_log_14 --> read_log_16
    read_log_16 --> read_log_17
    read_log_17 --> read_log_18
    read_log_18 --> read_log_19
    read_log_19 --> read_log_20
    read_log_20 --> read_log_21
    read_log_21 --> read_log_22
    read_log_22 --> read_log_23
    read_log_23 --> read_log_24
    read_log_24 --> read_log_25
    read_log_25 --> read_log_26
    read_log_26 --> read_log_27
    read_log_27 --> read_log_28
    read_log_28 --> read_log_29
    read_log_29 --> read_log_30
    read_log_30 --> read_log_31
    read_log_31 --> read_log_32
    read_log_32 --> read_log_33
    read_log_33 --> read_log_34
    read_log_34 --> read_log_35
    read_log_35 --> read_log_36
    read_log_36 --> read_log_37
    read_log_37 --> read_log_38
    read_log_38 --> read_log_39
    read_log_39 --> read_log_40
    read_log_40 --> read_log_41
    read_log_41 --> read_log_42
    read_log_42 --> read_log_43
    read_log_43 --> read_log_44
    read_log_44 --> read_log_45
    read_log_45 --> read_log_12
    read_log_12 --> read_log_46
    read_log_46 --> read_log_47
    read_log_47 --> read_log_48
```

## Function: `read_u32`

- File: MMSB/src/01_page/page.rs
- Branches: 1
- Loops: 0
- Nodes: 8
- Edges: 8

```mermaid
flowchart TD
    read_u32_0["ENTRY"]
    read_u32_1["if * cursor + 4 > blob . len ()"]
    read_u32_2["return Err (PageError :: MetadataDecode ('unexpected end of metadata'))"]
    read_u32_3["if join"]
    read_u32_4["let bytes : [u8 ; 4] = blob [* cursor .. * cursor + 4] . try_into () . map_err (| _ | PageError :: M..."]
    read_u32_5["* cursor += 4"]
    read_u32_6["Ok (u32 :: from_le_bytes (bytes))"]
    read_u32_7["EXIT"]
    read_u32_0 --> read_u32_1
    read_u32_1 --> read_u32_2
    read_u32_2 --> read_u32_3
    read_u32_1 --> read_u32_3
    read_u32_3 --> read_u32_4
    read_u32_4 --> read_u32_5
    read_u32_5 --> read_u32_6
    read_u32_6 --> read_u32_7
```

## Function: `serialize_frame`

- File: MMSB/src/01_page/tlog.rs
- Branches: 1
- Loops: 1
- Nodes: 24
- Edges: 25

```mermaid
flowchart TD
    serialize_frame_0["ENTRY"]
    serialize_frame_1["writer . write_all (& delta . delta_id . 0 . to_le_bytes ()) ?"]
    serialize_frame_2["writer . write_all (& delta . page_id . 0 . to_le_bytes ()) ?"]
    serialize_frame_3["writer . write_all (& delta . epoch . 0 . to_le_bytes ()) ?"]
    serialize_frame_4["let mask_len = delta . mask . len () as u32"]
    serialize_frame_5["writer . write_all (& mask_len . to_le_bytes ()) ?"]
    serialize_frame_6["for flag in & delta . mask"]
    serialize_frame_7["writer . write_all (& [* flag as u8]) ?"]
    serialize_frame_8["after for"]
    serialize_frame_9["let payload_len = delta . payload . len () as u32"]
    serialize_frame_10["writer . write_all (& payload_len . to_le_bytes ()) ?"]
    serialize_frame_11["writer . write_all (& delta . payload) ?"]
    serialize_frame_12["writer . write_all (& [delta . is_sparse as u8]) ?"]
    serialize_frame_13["writer . write_all (& delta . timestamp . to_le_bytes ()) ?"]
    serialize_frame_14["let source_bytes = delta . source . 0 . as_bytes ()"]
    serialize_frame_15["writer . write_all (& (source_bytes . len () as u32) . to_le_bytes ()) ?"]
    serialize_frame_16["writer . write_all (source_bytes) ?"]
    serialize_frame_17["let metadata_len = delta . intent_metadata . as_ref () . map (| s | s . as_bytes () . len () as ..."]
    serialize_frame_18["writer . write_all (& metadata_len . to_le_bytes ()) ?"]
    serialize_frame_19["if let Some (metadata) = & delta . intent_metadata"]
    serialize_frame_20["writer . write_all (metadata . as_bytes ()) ?"]
    serialize_frame_21["if join"]
    serialize_frame_22["Ok (())"]
    serialize_frame_23["EXIT"]
    serialize_frame_0 --> serialize_frame_1
    serialize_frame_1 --> serialize_frame_2
    serialize_frame_2 --> serialize_frame_3
    serialize_frame_3 --> serialize_frame_4
    serialize_frame_4 --> serialize_frame_5
    serialize_frame_5 --> serialize_frame_6
    serialize_frame_6 --> serialize_frame_7
    serialize_frame_7 --> serialize_frame_6
    serialize_frame_6 --> serialize_frame_8
    serialize_frame_8 --> serialize_frame_9
    serialize_frame_9 --> serialize_frame_10
    serialize_frame_10 --> serialize_frame_11
    serialize_frame_11 --> serialize_frame_12
    serialize_frame_12 --> serialize_frame_13
    serialize_frame_13 --> serialize_frame_14
    serialize_frame_14 --> serialize_frame_15
    serialize_frame_15 --> serialize_frame_16
    serialize_frame_16 --> serialize_frame_17
    serialize_frame_17 --> serialize_frame_18
    serialize_frame_18 --> serialize_frame_19
    serialize_frame_19 --> serialize_frame_20
    serialize_frame_20 --> serialize_frame_21
    serialize_frame_19 --> serialize_frame_21
    serialize_frame_21 --> serialize_frame_22
    serialize_frame_22 --> serialize_frame_23
```

## Function: `summary`

- File: MMSB/src/01_page/tlog.rs
- Branches: 1
- Loops: 1
- Nodes: 16
- Edges: 17

```mermaid
flowchart TD
    summary_0["ENTRY"]
    summary_1["let file = match File :: open (path . as_ref ()) { Ok (file) => file , Err (err) if err ..."]
    summary_2["if file . metadata () ? . len () == 0"]
    summary_3["return Ok (LogSummary :: default ())"]
    summary_4["if join"]
    summary_5["let mut reader = BufReader :: new (file)"]
    summary_6["let version = validate_header (& mut reader) ?"]
    summary_7["let mut summary = LogSummary :: default ()"]
    summary_8["while let Ok (Some (delta)) = read_frame (& mut reader , version)"]
    summary_9["summary . total_deltas += 1"]
    summary_10["let metadata_bytes = delta . intent_metadata . as_ref () . map (| m | m . as_bytes () . len () as ..."]
    summary_11["summary . total_bytes += delta . mask . len () as u64 + delta . payload . len..."]
    summary_12["summary . last_epoch = summary . last_epoch . max (delta . epoch . 0)"]
    summary_13["after while"]
    summary_14["Ok (summary)"]
    summary_15["EXIT"]
    summary_0 --> summary_1
    summary_1 --> summary_2
    summary_2 --> summary_3
    summary_3 --> summary_4
    summary_2 --> summary_4
    summary_4 --> summary_5
    summary_5 --> summary_6
    summary_6 --> summary_7
    summary_7 --> summary_8
    summary_8 --> summary_9
    summary_9 --> summary_10
    summary_10 --> summary_11
    summary_11 --> summary_12
    summary_12 --> summary_8
    summary_8 --> summary_13
    summary_13 --> summary_14
    summary_14 --> summary_15
```

## Function: `validate_delta`

- File: MMSB/src/01_page/delta_validation.rs
- Branches: 3
- Loops: 0
- Nodes: 12
- Edges: 14

```mermaid
flowchart TD
    validate_delta_0["ENTRY"]
    validate_delta_1["if delta . is_sparse"]
    validate_delta_2["let changed = delta . mask . iter () . filter (| & & bit | bit) . count ()"]
    validate_delta_3["if changed != delta . payload . len ()"]
    validate_delta_4["return Err (DeltaError :: SizeMismatch { mask_len : changed , payload_len : delta . ..."]
    validate_delta_5["if join"]
    validate_delta_6["if delta . mask . len () != delta . payload . len ()"]
    validate_delta_7["return Err (DeltaError :: SizeMismatch { mask_len : delta . mask . len () , payload_..."]
    validate_delta_8["if join"]
    validate_delta_9["if join"]
    validate_delta_10["Ok (())"]
    validate_delta_11["EXIT"]
    validate_delta_0 --> validate_delta_1
    validate_delta_1 --> validate_delta_2
    validate_delta_2 --> validate_delta_3
    validate_delta_3 --> validate_delta_4
    validate_delta_4 --> validate_delta_5
    validate_delta_3 --> validate_delta_5
    validate_delta_1 --> validate_delta_6
    validate_delta_6 --> validate_delta_7
    validate_delta_7 --> validate_delta_8
    validate_delta_6 --> validate_delta_8
    validate_delta_5 --> validate_delta_9
    validate_delta_8 --> validate_delta_9
    validate_delta_9 --> validate_delta_10
    validate_delta_10 --> validate_delta_11
```

## Function: `validate_header`

- File: MMSB/src/01_page/tlog.rs
- Branches: 2
- Loops: 0
- Nodes: 15
- Edges: 16

```mermaid
flowchart TD
    validate_header_0["ENTRY"]
    validate_header_1["reader . seek (SeekFrom :: Start (0)) ?"]
    validate_header_2["let mut magic = [0u8 ; 8]"]
    validate_header_3["reader . read_exact (& mut magic) ?"]
    validate_header_4["if & magic != MAGIC"]
    validate_header_5["return Err (std :: io :: Error :: new (std :: io :: ErrorKind :: InvalidData , 'inva..."]
    validate_header_6["if join"]
    validate_header_7["let mut version_bytes = [0u8 ; 4]"]
    validate_header_8["reader . read_exact (& mut version_bytes) ?"]
    validate_header_9["let version = u32 :: from_le_bytes (version_bytes)"]
    validate_header_10["if version < 1 || version > VERSION"]
    validate_header_11["return Err (std :: io :: Error :: new (std :: io :: ErrorKind :: InvalidData , 'unsu..."]
    validate_header_12["if join"]
    validate_header_13["Ok (version)"]
    validate_header_14["EXIT"]
    validate_header_0 --> validate_header_1
    validate_header_1 --> validate_header_2
    validate_header_2 --> validate_header_3
    validate_header_3 --> validate_header_4
    validate_header_4 --> validate_header_5
    validate_header_5 --> validate_header_6
    validate_header_4 --> validate_header_6
    validate_header_6 --> validate_header_7
    validate_header_7 --> validate_header_8
    validate_header_8 --> validate_header_9
    validate_header_9 --> validate_header_10
    validate_header_10 --> validate_header_11
    validate_header_11 --> validate_header_12
    validate_header_10 --> validate_header_12
    validate_header_12 --> validate_header_13
    validate_header_13 --> validate_header_14
```

## Function: `write_checkpoint`

- File: MMSB/src/01_page/checkpoint.rs
- Branches: 0
- Loops: 1
- Nodes: 21
- Edges: 21

```mermaid
flowchart TD
    write_checkpoint_0["ENTRY"]
    write_checkpoint_1["let pages = allocator . snapshot_pages ()"]
    write_checkpoint_2["let log_offset = tlog . current_offset () ?"]
    write_checkpoint_3["let mut writer = BufWriter :: new (File :: create (path) ?)"]
    write_checkpoint_4["writer . write_all (SNAPSHOT_MAGIC) ?"]
    write_checkpoint_5["writer . write_all (& SNAPSHOT_VERSION . to_le_bytes ()) ?"]
    write_checkpoint_6["writer . write_all (& (pages . len () as u32) . to_le_bytes ()) ?"]
    write_checkpoint_7["writer . write_all (& log_offset . to_le_bytes ()) ?"]
    write_checkpoint_8["for page in pages"]
    write_checkpoint_9["writer . write_all (& page . page_id . 0 . to_le_bytes ()) ?"]
    write_checkpoint_10["writer . write_all (& (page . size as u64) . to_le_bytes ()) ?"]
    write_checkpoint_11["writer . write_all (& page . epoch . to_le_bytes ()) ?"]
    write_checkpoint_12["writer . write_all (& (page . location as i32) . to_le_bytes ()) ?"]
    write_checkpoint_13["writer . write_all (& (page . metadata_blob . len () as u32) . to_le_bytes ()) ?"]
    write_checkpoint_14["writer . write_all (& page . metadata_blob) ?"]
    write_checkpoint_15["writer . write_all (& (page . data . len () as u32) . to_le_bytes ()) ?"]
    write_checkpoint_16["writer . write_all (& page . data) ?"]
    write_checkpoint_17["after for"]
    write_checkpoint_18["writer . flush () ?"]
    write_checkpoint_19["Ok (())"]
    write_checkpoint_20["EXIT"]
    write_checkpoint_0 --> write_checkpoint_1
    write_checkpoint_1 --> write_checkpoint_2
    write_checkpoint_2 --> write_checkpoint_3
    write_checkpoint_3 --> write_checkpoint_4
    write_checkpoint_4 --> write_checkpoint_5
    write_checkpoint_5 --> write_checkpoint_6
    write_checkpoint_6 --> write_checkpoint_7
    write_checkpoint_7 --> write_checkpoint_8
    write_checkpoint_8 --> write_checkpoint_9
    write_checkpoint_9 --> write_checkpoint_10
    write_checkpoint_10 --> write_checkpoint_11
    write_checkpoint_11 --> write_checkpoint_12
    write_checkpoint_12 --> write_checkpoint_13
    write_checkpoint_13 --> write_checkpoint_14
    write_checkpoint_14 --> write_checkpoint_15
    write_checkpoint_15 --> write_checkpoint_16
    write_checkpoint_16 --> write_checkpoint_8
    write_checkpoint_8 --> write_checkpoint_17
    write_checkpoint_17 --> write_checkpoint_18
    write_checkpoint_18 --> write_checkpoint_19
    write_checkpoint_19 --> write_checkpoint_20
```

