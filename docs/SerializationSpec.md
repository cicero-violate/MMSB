# MMSB Serialization Specification

All serialized artifacts obey a single rule: `state × delta → state′` must hold even when the payload is replayed outside the original process. Formats are designed for deterministic rehydration and forward compatibility.

## Data Types
| Name | Julia Type | Description |
| --- | --- | --- |
| `PageID` | `UInt64` | Globally unique page handle. |
| `DeltaID` | `UInt64` | Globally unique delta handle. |
| `Epoch` | `UInt32` | Monotonic counter per page. |
| `Timestamp` | `UInt64` | Nanoseconds since epoch. |
| `PageLocation` | `Int32` | Enum tag (`0=CPU`, `1=GPU`, `2=UNIFIED`). |

## Page Payload (`serialize_page`)
Pages are serialized via Julia Serialization with the tuple:
```
(page_id::PageID,
 epoch::UInt32,
 packed_mask::Vector{UInt8},
 size::Int64,
 compressed_data::Vector{UInt8},
 location_tag::Int32,
 metadata::Dict{Symbol,Any})
```

1. **Mask packing** — `_pack_bool_vector` packs 8 bits per byte (little-endian within each byte).
2. **Data compression** — `_rle_compress` emits `(value::UInt8, run_length::UInt8)` pairs. `run_length` saturates at 255, so long runs split into multiple pairs.
3. **Metadata** — Deep-copied `Dict{Symbol,Any}` enabling structured annotations.
4. **GPU fallback** — During `deserialize_page`, GPU buffers degrade to CPU storage when CUDA is unavailable, ensuring deterministic behavior.

## Delta Payload (`serialize_delta`)
Also serialized via Julia Serialization:
```
(delta_id::DeltaID,
 page_id::PageID,
 epoch::UInt32,
 mask::Vector{Bool},
 payload::Vector{UInt8},
 is_sparse::Bool,
 timestamp::UInt64,
 source::Symbol)
```

- **Sparse encoding** — When `is_sparse=true`, `payload` length equals `count(mask)` and stores only changed bytes. Otherwise it equals `length(mask)` (dense).
- **Dense reconstruction** — `dense_data` copies the payload, filling zeros for untouched indices if sparse.
- **Timestamp** — Filled by the constructor for log queries (`query_log` filters by time window).

## Transaction Log (`checkpoint_log!`)
Binary layout written directly to `IO`:

| Order | Type | Description |
| --- | --- | --- |
| 1 | `String` | Magic header `"MMSBCHK1"`. |
| 2 | `UInt32` | Version (`1` as of Task 6.5). |
| 3 | `UInt64` | Capture timestamp (`time_ns()`). |
| 4 | `Int64` | Number of serialized pages (`page_count`). |
| `repeat page_count` | — | For each page: `PageID`, `Int64` byte length, raw bytes from `serialize_page`. |
| Next | `Int64` | Number of serialized deltas (`delta_count`). |
| `repeat delta_count` | — | For each delta: `Int64` byte length, raw bytes from `serialize_delta`. |

### Validation
- `load_checkpoint!` asserts the magic/version and raises `SerializationError` when corruption is detected (bad lengths, truncated RLE payloads, or unsupported versions).
- GPU pages are materialized on CPU if CUDA is missing to ensure replay still succeeds.

## Replay Contract
`replay_log(state, start_epoch, end_epoch)`:
1. Copies pages whose `epoch` lies within the requested window.
2. Accumulates deltas whose `epoch` lies within the same bounds.
3. Sorts deltas by epoch and applies masked writes in order.
4. Converts GPU pages to CPU buffers for deterministic snapshots.

The replay snapshot is detached from the live state, so it can be serialized again (e.g., for diffing) without mutating the running system.

## Compression + Compaction
- `_merge_page_deltas` groups deltas per `PageID`, reconstructs dense payloads, applies masked writes, and creates a new merged `Delta`. The merged delta inherits the latest epoch/source and receives a fresh `DeltaID`.
- `compress_log!` replaces the state’s `tlog` with the merged set, sorted by epoch.

## Compatibility Notes
- Always bump `CHECKPOINT_VERSION` when changing the binary layout. `load_checkpoint!` already checks `version <= CHECKPOINT_VERSION`.
- When extending metadata with new fields, prefer dictionary entries so older builds can ignore unknown keys after deserialization.
