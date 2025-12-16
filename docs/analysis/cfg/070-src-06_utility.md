# CFG Group: src/06_utility

## Function: `cpu_has_avx2`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_avx2_0["ENTRY"]
    cpu_has_avx2_1["CpuFeatures :: get () . avx2"]
    cpu_has_avx2_2["EXIT"]
    cpu_has_avx2_0 --> cpu_has_avx2_1
    cpu_has_avx2_1 --> cpu_has_avx2_2
```

## Function: `cpu_has_avx512`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_avx512_0["ENTRY"]
    cpu_has_avx512_1["CpuFeatures :: get () . avx512f"]
    cpu_has_avx512_2["EXIT"]
    cpu_has_avx512_0 --> cpu_has_avx512_1
    cpu_has_avx512_1 --> cpu_has_avx512_2
```

## Function: `cpu_has_sse42`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_sse42_0["ENTRY"]
    cpu_has_sse42_1["CpuFeatures :: get () . sse42"]
    cpu_has_sse42_2["EXIT"]
    cpu_has_sse42_0 --> cpu_has_sse42_1
    cpu_has_sse42_1 --> cpu_has_sse42_2
```

## Function: `test_cache_hit_rate`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_cache_hit_rate_0["ENTRY"]
    test_cache_hit_rate_1["let telemetry = Telemetry :: new ()"]
    test_cache_hit_rate_2["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_3["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_4["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_5["telemetry . record_cache_miss ()"]
    test_cache_hit_rate_6["let snapshot = telemetry . snapshot ()"]
    test_cache_hit_rate_7["macro assert_eq"]
    test_cache_hit_rate_8["EXIT"]
    test_cache_hit_rate_0 --> test_cache_hit_rate_1
    test_cache_hit_rate_1 --> test_cache_hit_rate_2
    test_cache_hit_rate_2 --> test_cache_hit_rate_3
    test_cache_hit_rate_3 --> test_cache_hit_rate_4
    test_cache_hit_rate_4 --> test_cache_hit_rate_5
    test_cache_hit_rate_5 --> test_cache_hit_rate_6
    test_cache_hit_rate_6 --> test_cache_hit_rate_7
    test_cache_hit_rate_7 --> test_cache_hit_rate_8
```

## Function: `test_reset`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_reset_0["ENTRY"]
    test_reset_1["let telemetry = Telemetry :: new ()"]
    test_reset_2["telemetry . record_cache_hit ()"]
    test_reset_3["telemetry . reset ()"]
    test_reset_4["let snapshot = telemetry . snapshot ()"]
    test_reset_5["macro assert_eq"]
    test_reset_6["EXIT"]
    test_reset_0 --> test_reset_1
    test_reset_1 --> test_reset_2
    test_reset_2 --> test_reset_3
    test_reset_3 --> test_reset_4
    test_reset_4 --> test_reset_5
    test_reset_5 --> test_reset_6
```

## Function: `test_telemetry_basic`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_telemetry_basic_0["ENTRY"]
    test_telemetry_basic_1["let telemetry = Telemetry :: new ()"]
    test_telemetry_basic_2["telemetry . record_cache_hit ()"]
    test_telemetry_basic_3["telemetry . record_cache_miss ()"]
    test_telemetry_basic_4["telemetry . record_allocation (4096)"]
    test_telemetry_basic_5["let snapshot = telemetry . snapshot ()"]
    test_telemetry_basic_6["macro assert_eq"]
    test_telemetry_basic_7["macro assert_eq"]
    test_telemetry_basic_8["macro assert_eq"]
    test_telemetry_basic_9["macro assert_eq"]
    test_telemetry_basic_10["EXIT"]
    test_telemetry_basic_0 --> test_telemetry_basic_1
    test_telemetry_basic_1 --> test_telemetry_basic_2
    test_telemetry_basic_2 --> test_telemetry_basic_3
    test_telemetry_basic_3 --> test_telemetry_basic_4
    test_telemetry_basic_4 --> test_telemetry_basic_5
    test_telemetry_basic_5 --> test_telemetry_basic_6
    test_telemetry_basic_6 --> test_telemetry_basic_7
    test_telemetry_basic_7 --> test_telemetry_basic_8
    test_telemetry_basic_8 --> test_telemetry_basic_9
    test_telemetry_basic_9 --> test_telemetry_basic_10
```

