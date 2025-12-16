# Structure Group: src/01_page

## File: MMSB/src/01_page/Delta.jl

- Layer(s): 01_page
- Language coverage: Julia (25)
- Element types: Function (22), Module (1), Struct (2)
- Total elements: 25

### Elements

- [Julia | Module] `DeltaTypes` (line 1, pub)
- [Julia | Struct] `Delta` (line 11, pub)
  - Signature: `mutable struct Delta`
- [Julia | Function] `Delta` (line 24, pub)
  - Signature: `Delta(id::DeltaID, page_id::PageID, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol`
- [Julia | Function] `Delta` (line 36, pub)
  - Signature: `Delta(handle::FFIWrapper.RustDeltaHandle)`
  - Calls: FFIWrapper.rust_delta_epoch, FFIWrapper.rust_delta_free!, FFIWrapper.rust_delta_id, FFIWrapper.rust_delta_intent_metadata, FFIWrapper.rust_delta_is_sparse, FFIWrapper.rust_delta_mask, FFIWrapper.rust_delta_page_id, FFIWrapper.rust_delta_payload, FFIWrapper.rust_delta_source, FFIWrapper.rust_delta_timestamp, PageID, Symbol, error, finalizer, new
- [Julia | Function] `new_delta_handle` (line 55, pub)
  - Signature: `new_delta_handle(id::DeltaID, page_id::PageID, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Sy...`
- [Julia | Function] `apply_delta!` (line 61, pub)
  - Signature: `apply_delta!(page_handle::FFIWrapper.RustPageHandle, delta::Delta)`
  - Calls: FFIWrapper.rust_delta_apply!
- [Julia | Function] `dense_data` (line 65, pub)
  - Signature: `dense_data(delta::Delta)::Vector{UInt8}`
  - Calls: copy, eachindex, length
- [Julia | Function] `serialize_delta` (line 82, pub)
  - Signature: `serialize_delta(delta::Delta)::Vector{UInt8}`
  - Calls: IOBuffer, Serialization.serialize, map, take!
- [Julia | Function] `deserialize_delta` (line 100, pub)
  - Signature: `deserialize_delta(bytes::Vector{UInt8})::Delta`
  - Calls: Delta, IOBuffer, Serialization.deserialize, UInt8.
- [Julia | Function] `set_intent_metadata!` (line 110, pub)
  - Signature: `set_intent_metadata!(delta::Delta, metadata::Union{Nothing,AbstractString,Dict{Symbol,Any}})`
  - Calls: ArgumentError, FFIWrapper.rust_delta_set_intent_metadata!, String, _encode_metadata_dict, throw, typeof
- [Julia | Function] `intent_metadata` (line 130, pub)
  - Signature: `intent_metadata(delta::Delta; parse::Bool`
- [Julia | Function] `_encode_metadata_value` (line 136, pub)
  - Signature: `_encode_metadata_value(value)`
  - Calls: Dict, String, Symbol, _encode_metadata_dict, _encode_metadata_value, _escape_metadata_string, join, string
- [Julia | Function] `_encode_metadata_dict` (line 160, pub)
  - Signature: `_encode_metadata_dict(metadata::AbstractDict)`
  - Calls: String, Symbol, _encode_metadata_value, _escape_metadata_string, join, push!, string
- [Julia | Function] `_escape_metadata_string` (line 169, pub)
  - Signature: `_escape_metadata_string(str::AbstractString)`
  - Calls: IOBuffer, String, print, take!
- [Julia | Function] `merge_deltas_simd!` (line 194, pub)
  - Signature: `merge_deltas_simd!( data_a::Vector{UInt8}, mask_a::Vector{Bool}, data_b::Vector{UInt8}, mask_b::Vector{Bool}, out_dat...`
  - Calls: ccall, length, min
- [Julia | Struct] `_MetadataParser` (line 219, pub)
  - Signature: `mutable struct _MetadataParser`
- [Julia | Function] `_decode_metadata` (line 224, pub)
  - Signature: `_decode_metadata(json::String)`
  - Calls: Dict, Symbol, _MetadataParser, _parse_metadata_value, error
- [Julia | Function] `_parse_metadata_value` (line 231, pub)
  - Signature: `_parse_metadata_value(parser::_MetadataParser)`
  - Calls: _parse_metadata_array, _parse_metadata_number, _parse_metadata_object, _parse_metadata_string, _peek, _skip_ws, error, startswith
- [Julia | Function] `_parse_metadata_object` (line 254, pub)
  - Signature: `_parse_metadata_object(parser::_MetadataParser)`
  - Calls: _consume, _parse_metadata_string, _parse_metadata_value, _peek, _skip_ws
- [Julia | Function] `_parse_metadata_array` (line 279, pub)
  - Signature: `_parse_metadata_array(parser::_MetadataParser)`
  - Calls: _consume, _parse_metadata_value, _peek, _skip_ws, push!
- [Julia | Function] `_parse_metadata_string` (line 300, pub)
  - Signature: `_parse_metadata_string(parser::_MetadataParser)`
  - Calls: IOBuffer, String, _consume, error, lastindex, print, take!
- [Julia | Function] `_parse_metadata_number` (line 329, pub)
  - Signature: `_parse_metadata_number(parser::_MetadataParser)`
  - Calls: error, lastindex, lowercase, occursin, parse, tryparse
- [Julia | Function] `_skip_ws` (line 348, pub)
  - Signature: `_skip_ws(parser::_MetadataParser)`
  - Calls: in, lastindex
- [Julia | Function] `_consume` (line 354, pub)
  - Signature: `_consume(parser::_MetadataParser, expected::Char)`
  - Calls: _peek, error
- [Julia | Function] `_peek` (line 360, pub)
  - Signature: `_peek(parser::_MetadataParser)`
  - Calls: error, lastindex

## File: MMSB/src/01_page/Page.jl

- Layer(s): 01_page
- Language coverage: Julia (15)
- Element types: Function (13), Module (1), Struct (1)
- Total elements: 15

### Elements

- [Julia | Module] `PageTypes` (line 1, pub)
- [Julia | Struct] `Page` (line 17, pub)
  - Signature: `mutable struct Page`
- [Julia | Function] `Page` (line 27, pub)
  - Signature: `Page(handle::FFIWrapper.RustPageHandle, id::PageID, location::PageLocation, size::Int; metadata::Dict{Symbol,Any}`
- [Julia | Function] `is_gpu_page` (line 42, pub)
  - Signature: `is_gpu_page(page::Page)`
  - Calls: in
- [Julia | Function] `is_cpu_page` (line 43, pub)
  - Signature: `is_cpu_page(page::Page)`
- [Julia | Function] `page_size_bytes` (line 44, pub)
  - Signature: `page_size_bytes(page::Page)`
- [Julia | Function] `initialize!` (line 46, pub)
  - Signature: `initialize!(page::Page)`
- [Julia | Function] `activate!` (line 52, pub)
  - Signature: `activate!(page::Page)`
- [Julia | Function] `deactivate!` (line 58, pub)
  - Signature: `deactivate!(page::Page)`
- [Julia | Function] `read_page` (line 64, pub)
  - Signature: `read_page(page::Page)::Vector{UInt8}`
  - Calls: FFIWrapper.rust_page_read!
- [Julia | Function] `_apply_metadata!` (line 73, pub)
  - Signature: `_apply_metadata!(page::Page, metadata::Dict{Symbol,Any})`
  - Calls: FFIWrapper.rust_page_metadata_import!, _encode_metadata_dict, isempty
- [Julia | Function] `_encode_metadata_dict` (line 81, pub)
  - Signature: `_encode_metadata_dict(metadata::Dict{Symbol,Any})`
  - Calls: IOBuffer, String, UInt32, _coerce_metadata_value, codeunits, length, take!, write
- [Julia | Function] `_coerce_metadata_value` (line 96, pub)
  - Signature: `_coerce_metadata_value(value::Any)`
  - Calls: ArgumentError, String, codeunits, throw
- [Julia | Function] `_decode_metadata_blob` (line 102, pub)
  - Signature: `_decode_metadata_blob(blob::Vector{UInt8})`
  - Calls: IOBuffer, String, Symbol, isempty, read
- [Julia | Function] `metadata_from_blob` (line 117, pub)
  - Signature: `metadata_from_blob(blob::Vector{UInt8})`
  - Calls: _decode_metadata_blob

## File: MMSB/src/01_page/ReplayEngine.jl

- Layer(s): 01_page
- Language coverage: Julia (12)
- Element types: Function (11), Module (1)
- Total elements: 12

### Elements

- [Julia | Module] `ReplayEngine` (line 1, pub)
- [Julia | Function] `_blank_state_like` (line 13, pub)
  - Signature: `_blank_state_like(state::MMSBState)::MMSBState`
  - Calls: FFIWrapper.rust_allocator_allocate, Int32, MMSBConfig, MMSBState, Page, UInt64, activate!, initialize!, lock, register_page!
- [Julia | Function] `_apply_delta!` (line 48, pub)
  - Signature: `_apply_delta!(page::Page, delta::Delta)`
  - Calls: FFIWrapper.rust_delta_apply!
- [Julia | Function] `_all_deltas` (line 54, pub)
  - Signature: `_all_deltas(state::MMSBState)`
  - Calls: TLog.query_log
- [Julia | Function] `replay_to_epoch` (line 58, pub)
  - Signature: `replay_to_epoch(state::MMSBState, target_epoch::UInt32)::MMSBState`
  - Calls: _all_deltas, _apply_delta!, _blank_state_like, get_page
- [Julia | Function] `replay_to_timestamp` (line 71, pub)
  - Signature: `replay_to_timestamp(state::MMSBState, target_time::UInt64)::MMSBState`
  - Calls: _all_deltas, _apply_delta!, _blank_state_like, get_page
- [Julia | Function] `replay_from_checkpoint` (line 84, pub)
  - Signature: `replay_from_checkpoint(path::AbstractString, target_epoch::Union{UInt32, Nothing}`
- [Julia | Function] `replay_page_history` (line 92, pub)
  - Signature: `replay_page_history(state::MMSBState, page_id::PageID)::Vector{Vector{UInt8}}`
  - Calls: FFIWrapper.rust_page_epoch, FFIWrapper.rust_page_write_masked!, Page, TLog.query_log, UInt8, _apply_delta!, activate!, fill, get_page, initialize!, push!, read_page
- [Julia | Function] `verify_state_consistency` (line 115, pub)
  - Signature: `verify_state_consistency(state::MMSBState)::Bool`
  - Calls: get_page, lock, read_page, replay_to_epoch, typemax
- [Julia | Function] `replay_with_predicate` (line 129, pub)
  - Signature: `replay_with_predicate(state::MMSBState, predicate::Function)`
  - Calls: _all_deltas, predicate, push!
- [Julia | Function] `incremental_replay!` (line 137, pub)
  - Signature: `incremental_replay!(state::MMSBState, deltas::Vector{Delta})`
  - Calls: _apply_delta!, get_page
- [Julia | Function] `compute_diff` (line 146, pub)
  - Signature: `compute_diff(::MMSBState, ::MMSBState)`
  - Calls: error

## File: MMSB/src/01_page/TLog.jl

- Layer(s): 01_page
- Language coverage: Julia (14)
- Element types: Function (13), Module (1)
- Total elements: 14

### Elements

- [Julia | Module] `TLog` (line 1, pub)
- [Julia | Function] `compress_delta_mask` (line 19, pub)
  - Signature: `compress_delta_mask(mask::Vector{Bool}, mode::CompressionMode)`
  - Calls: UInt8., return
- [Julia | Function] `_with_rust_errors` (line 28, pub)
  - Signature: `_with_rust_errors(f::Function, context::String)`
  - Calls: f, rethrow, rethrow_translated
- [Julia | Function] `append_to_log!` (line 39, pub)
  - Signature: `append_to_log!(state::MMSBState, delta::Delta)`
  - Calls: FFIWrapper.rust_tlog_append!, _with_rust_errors
- [Julia | Function] `log_summary` (line 48, pub)
  - Signature: `log_summary(state::MMSBState)`
  - Calls: FFIWrapper.rust_tlog_summary, _with_rust_errors
- [Julia | Function] `_iterate_log` (line 56, pub)
  - Signature: `_iterate_log(f::Function, path::AbstractString)`
  - Calls: FFIWrapper.rust_tlog_reader_free!, FFIWrapper.rust_tlog_reader_new, f
- [Julia | Function] `query_log` (line 66, pub)
  - Signature: `query_log(state::MMSBState; page_id::Union{PageID,Nothing}`
- [Julia | Function] `get_deltas_for_page` (line 95, pub)
  - Signature: `get_deltas_for_page(state::MMSBState, pid::PageID)`
  - Calls: query_log
- [Julia | Function] `get_deltas_in_range` (line 97, pub)
  - Signature: `get_deltas_in_range(state::MMSBState, start_idx::Int, end_idx::Int)`
  - Calls: clamp, isempty, length, query_log
- [Julia | Function] `compute_log_statistics` (line 105, pub)
  - Signature: `compute_log_statistics(state::MMSBState)`
  - Calls: Dict, log_summary
- [Julia | Function] `replay_log` (line 114, pub)
  - Signature: `replay_log(state::MMSBState, target_epoch::UInt32)`
  - Calls: engine.replay_to_epoch, getfield, parentmodule
- [Julia | Function] `checkpoint_log!` (line 120, pub)
  - Signature: `checkpoint_log!(state::MMSBState, path::AbstractString)`
  - Calls: FFIWrapper.rust_checkpoint_write!, _with_rust_errors
- [Julia | Function] `load_checkpoint!` (line 128, pub)
  - Signature: `load_checkpoint!(state::MMSBState, path::AbstractString)`
  - Calls: FFIWrapper.rust_checkpoint_load!, _with_rust_errors
- [Julia | Function] `_refresh_pages!` (line 137, pub)
  - Signature: `_refresh_pages!(state::MMSBState)`
  - Calls: FFIWrapper.rust_allocator_acquire_page, FFIWrapper.rust_allocator_page_infos, Int, Page, PageID, PageLocation, activate!, codeunits, empty!, initialize!, lock, metadata_from_blob, unsafe_string

## File: MMSB/src/01_page/checkpoint.rs

- Layer(s): 01_page
- Language coverage: Rust (2)
- Element types: Function (2)
- Total elements: 2

### Elements

- [Rust | Function] `load_checkpoint` (line 0, pub)
  - Signature: `pub fn load_checkpoint (allocator : & PageAllocator , _tlog : & TransactionLog , path : impl AsRef < Path > ,) -> std...`
  - Calls: BufReader::new, File::open, read_exact, Err, std::io::Error::new, read_exact, u32::from_le_bytes, Err, std::io::Error::new, read_exact, u32::from_le_bytes, read_exact, Vec::with_capacity, read_exact, PageID, u64::from_le_bytes, read_exact, u64::from_le_bytes, read_exact, u32::from_le_bytes, read_exact, i32::from_le_bytes, map_err, PageLocation::from_tag, std::io::Error::new, read_exact, u32::from_le_bytes, read_exact, read_exact, u32::from_le_bytes, read_exact, push, restore_from_snapshot, Ok, Err, std::io::Error::new
- [Rust | Function] `write_checkpoint` (line 0, pub)
  - Signature: `pub fn write_checkpoint (allocator : & PageAllocator , tlog : & TransactionLog , path : impl AsRef < Path > ,) -> std...`
  - Calls: snapshot_pages, current_offset, BufWriter::new, File::create, write_all, write_all, to_le_bytes, write_all, to_le_bytes, len, write_all, to_le_bytes, write_all, to_le_bytes, write_all, to_le_bytes, write_all, to_le_bytes, write_all, to_le_bytes, write_all, to_le_bytes, len, write_all, write_all, to_le_bytes, len, write_all, flush, Ok

## File: MMSB/src/01_page/delta.rs

- Layer(s): 01_page
- Language coverage: Rust (6)
- Element types: Enum (1), Function (1), Impl (1), Struct (3)
- Total elements: 6

### Elements

- [Rust | Struct] `Delta` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct Delta { pub delta_id : DeltaID , pub page_id : PageID , pub epoch : Epoch , pub...`
- [Rust | Enum] `DeltaError` (line 0, pub)
- [Rust | Struct] `DeltaID` (line 0, pub)
  - Signature: `# [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq , Eq , Hash)] pub struct DeltaID (pub u64) ;`
- [Rust | Struct] `Source` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct Source (pub String) ;`
- [Rust | Impl] `impl Delta { pub fn new_dense (delta_id : DeltaID , page_id : PageID , epoch : Epoch , data : Vec < u8 > , mask : Vec < bool > , source : Source ,) -> Result < Self , DeltaError > { if data . len () != mask . len () { return Err (DeltaError :: SizeMismatch { mask_len : mask . len () , payload_len : data . len () , }) ; } Ok (Self { delta_id , page_id , epoch , mask , payload : data , is_sparse : false , timestamp : now_ns () , source , intent_metadata : None , }) } pub fn new_sparse (delta_id : DeltaID , page_id : PageID , epoch : Epoch , mask : Vec < bool > , payload : Vec < u8 > , source : Source ,) -> Result < Self , DeltaError > { let changed = mask . iter () . filter (| & & m | m) . count () ; if changed != payload . len () { return Err (DeltaError :: SizeMismatch { mask_len : mask . len () , payload_len : payload . len () , }) ; } Ok (Self { delta_id , page_id , epoch , mask , payload , is_sparse : true , timestamp : now_ns () , source , intent_metadata : None , }) } pub fn merge (& self , other : & Delta) -> Result < Delta , DeltaError > { if self . page_id != other . page_id { return Err (DeltaError :: PageIDMismatch { expected : self . page_id , found : other . page_id , }) ; } if self . mask . len () != other . mask . len () { return Err (DeltaError :: MaskSizeMismatch { expected : self . mask . len () , found : other . mask . len () , }) ; } let mut merged_mask = self . mask . clone () ; let mut merged_payload = self . to_dense () ; let other_dense = other . to_dense () ; for (idx , & flag) in other . mask . iter () . enumerate () { if flag { merged_mask [idx] = true ; merged_payload [idx] = other_dense [idx] ; } } Ok (Delta { delta_id : other . delta_id , page_id : other . page_id , epoch : Epoch (other . epoch . 0 . max (self . epoch . 0)) , mask : merged_mask , payload : merged_payload , is_sparse : false , timestamp : other . timestamp . max (self . timestamp) , source : other . source . clone () , intent_metadata : other . intent_metadata . clone () . or_else (| | self . intent_metadata . clone ()) , }) } pub fn to_dense (& self) -> Vec < u8 > { if ! self . is_sparse { return self . payload . clone () ; } let mut dense = vec ! [0u8 ; self . mask . len ()] ; let mut payload_idx = 0 ; for (idx , & flag) in self . mask . iter () . enumerate () { if flag { dense [idx] = self . payload [payload_idx] ; payload_idx += 1 ; } } dense } pub fn apply_to (& self , page : & mut Page) -> Result < () , PageError > { if let Err (err) = super :: delta_validation :: validate_delta (self) { return Err (match err { DeltaError :: SizeMismatch { mask_len , payload_len } => PageError :: MaskSizeMismatch { expected : mask_len , found : payload_len , } , DeltaError :: PageIDMismatch { expected , found } => PageError :: PageIDMismatch { expected , found , } , DeltaError :: MaskSizeMismatch { expected , found } => PageError :: MaskSizeMismatch { expected , found , } , }) ; } page . apply_delta (self) } } . self_ty` (line 0, priv)
- [Rust | Function] `now_ns` (line 0, priv)
  - Signature: `fn now_ns () -> u64 { SystemTime :: now () . duration_since (UNIX_EPOCH) . unwrap_or_default () . as_nanos () as u64 ...`
  - Calls: as_nanos, unwrap_or_default, duration_since, SystemTime::now

## File: MMSB/src/01_page/delta_merge.rs

- Layer(s): 01_page
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `merge_deltas` (line 0, pub)
  - Signature: `# [doc = " Merge deltas using SIMD when available"] pub fn merge_deltas (first : & Delta , second : & Delta) -> Resul...`
  - Calls: merge
- [Rust | Function] `merge_dense_avx2` (line 0, priv)
  - Signature: `# [doc = " SIMD-optimized dense delta merge using AVX2"] # [cfg (target_arch = "x86_64")] # [target_feature (enable =...`
  - Calls: min, len, len, _mm256_loadu_si256, add, as_ptr, _mm256_loadu_si256, add, as_ptr, _mm256_loadu_si256, as_ptr, _mm256_loadu_si256, as_ptr, _mm256_blendv_epi8, _mm256_storeu_si256, add, as_mut_ptr, _mm256_or_si256, _mm256_storeu_si256, as_mut_ptr
- [Rust | Function] `merge_dense_avx512` (line 0, priv)
  - Signature: `# [doc = " SIMD-optimized dense delta merge using AVX-512"] # [cfg (target_arch = "x86_64")] # [target_feature (enabl...`
  - Calls: min, len, len, _mm512_loadu_si512, add, as_ptr, _mm512_loadu_si512, add, as_ptr, _mm512_loadu_si512, as_ptr, _mm512_test_epi8_mask, _mm512_mask_blend_epi8, _mm512_storeu_si512, add, as_mut_ptr
- [Rust | Function] `merge_dense_simd` (line 0, pub)
  - Signature: `# [doc = " Dispatch to appropriate SIMD implementation"] pub fn merge_dense_simd (data_a : & [u8] , mask_a : & [bool]...`
  - Calls: merge_dense_avx512, merge_dense_avx2, min, len, len

## File: MMSB/src/01_page/delta_validation.rs

- Layer(s): 01_page
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `validate_delta` (line 0, pub)
  - Signature: `# [doc = " Validate structural consistency of a delta before application."] pub fn validate_delta (delta : & Delta) -...`
  - Calls: count, filter, iter, len, Err, len, len, len, Err, len, len, Ok

## File: MMSB/src/01_page/epoch.rs

- Layer(s): 01_page
- Language coverage: Rust (4)
- Element types: Impl (2), Struct (2)
- Total elements: 4

### Elements

- [Rust | Struct] `Epoch` (line 0, pub)
  - Signature: `# [doc = " Page epoch (monotonic counter)"] # [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq , Eq ,...`
- [Rust | Struct] `EpochCell` (line 0, pub)
  - Signature: `# [doc = " Thread-safe epoch cell reused inside the Rust Page structure."] # [derive (Debug)] pub struct EpochCell { ...`
- [Rust | Impl] `impl Epoch { pub fn new (value : u32) -> Self { Epoch (value) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl EpochCell { pub fn new (value : u32) -> Self { Self { inner : AtomicU32 :: new (value) , } } # [inline] pub fn load (& self) -> Epoch { Epoch (self . inner . load (Ordering :: Acquire)) } # [inline] pub fn store (& self , value : Epoch) { self . inner . store (value . 0 , Ordering :: Release) ; } # [inline] pub fn increment (& self) -> Epoch { let old = self . inner . fetch_add (1 , Ordering :: AcqRel) ; println ! ("EPOCH_INCREMENT: was {} → now {}" , old , old + 1) ; Epoch (old) } } . self_ty` (line 0, priv)

## File: MMSB/src/01_page/mod.rs

- Layer(s): 01_page
- Language coverage: Rust (11)
- Element types: Module (11)
- Total elements: 11

### Elements

- [Rust | Module] `checkpoint` (line 0, pub)
- [Rust | Module] `delta` (line 0, pub)
- [Rust | Module] `delta_merge` (line 0, pub)
- [Rust | Module] `delta_validation` (line 0, pub)
- [Rust | Module] `epoch` (line 0, pub)
- [Rust | Module] `page` (line 0, pub)
- [Rust | Module] `simd_mask` (line 0, pub)
- [Rust | Module] `tlog` (line 0, pub)
- [Rust | Module] `tlog_compression` (line 0, pub)
- [Rust | Module] `tlog_replay` (line 0, pub)
- [Rust | Module] `tlog_serialization` (line 0, pub)

## File: MMSB/src/01_page/page.rs

- Layer(s): 01_page
- Language coverage: Rust (16)
- Element types: Enum (2), Function (3), Impl (8), Struct (3)
- Total elements: 16

### Elements

- [Rust | Impl] `Clone for impl Clone for Page { fn clone (& self) -> Self { let new_debug_id = PAGE_COUNTER . fetch_add (1 , Ordering :: Relaxed) ; println ! ("[PAGE {:>4}] CLONE → [PAGE {:>4}]  id={:>6}" , self . debug_id , new_debug_id , self . id . 0) ; let layout_data = std :: alloc :: Layout :: array :: < u8 > (self . capacity) . expect ("invalid capacity") ; let data = unsafe { std :: alloc :: alloc (layout_data) } ; unsafe { std :: ptr :: copy_nonoverlapping (self . data , data , self . capacity) ; } let mask_size = (self . capacity + 7) / 8 ; let layout_mask = std :: alloc :: Layout :: array :: < u8 > (mask_size) . expect ("invalid mask size") ; let mask = unsafe { std :: alloc :: alloc (layout_mask) } ; unsafe { std :: ptr :: copy_nonoverlapping (self . mask , mask , mask_size) ; } Self { debug_id : new_debug_id , id : self . id , epoch : EpochCell :: new (self . epoch . load () . 0) , data , mask , capacity : self . capacity , location : self . location , metadata : self . metadata . clone () , unified_cuda_backing : false , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `Display for impl fmt :: Display for PageID { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { write ! (f , "{}" , self . 0) } } . self_ty` (line 0, priv)
- [Rust | Impl] `Drop for impl Drop for Page { fn drop (& mut self) { println ! ("[PAGE {:>4}] DROP    id={:>6} loc={:?}" , self . debug_id , self . id . 0 , self . location) ; let mask_size = (self . capacity + 7) / 8 ; let mask_layout = std :: alloc :: Layout :: array :: < u8 > (mask_size) . unwrap () ; unsafe { std :: alloc :: dealloc (self . mask , mask_layout) } ; if self . location == PageLocation :: Unified && self . unified_cuda_backing { # [cfg (feature = "cuda")] unsafe { let _ = cudaFree (self . data as * mut c_void) ; } } else { unsafe { let layout = std :: alloc :: Layout :: array :: < u8 > (self . capacity) . unwrap () ; std :: alloc :: dealloc (self . data , layout) ; } } } } . self_ty` (line 0, priv)
- [Rust | Struct] `Metadata` (line 0, pub)
  - Signature: `# [doc = " Metadata key-value store with copy-on-write semantics."] # [derive (Debug , Clone , Default)] pub struct M...`
- [Rust | Struct] `Page` (line 0, pub)
  - Signature: `# [doc = " Memory page implementation shared across the runtime layers."] # [derive (Debug)] pub struct Page { debug_...`
- [Rust | Enum] `PageError` (line 0, pub)
- [Rust | Struct] `PageID` (line 0, pub)
  - Signature: `# [doc = " Globally unique identifier for pages."] # [repr (transparent)] # [derive (Debug , Clone , Copy , PartialEq...`
- [Rust | Enum] `PageLocation` (line 0, pub)
- [Rust | Impl] `Send for unsafe impl Send for Page { } . self_ty` (line 0, priv)
- [Rust | Impl] `Sync for unsafe impl Sync for Page { } . self_ty` (line 0, priv)
- [Rust | Function] `allocate_zeroed` (line 0, priv)
  - Signature: `fn allocate_zeroed (size : usize , err_code : i32) -> Result < * mut u8 , PageError > { let layout = std :: alloc :: ...`
  - Calls: map_err, std::alloc::Layout::array, PageError::AllocError, std::alloc::alloc_zeroed, is_null, Err, PageError::AllocError, Ok
- [Rust | Impl] `impl Metadata { pub fn new () -> Self { Self { store : Arc :: new (RwLock :: new (Vec :: new ())) , } } pub fn insert (& self , key : impl Into < String > , value : Vec < u8 >) { let key_string = key . into () ; let mut guard = self . store . write () ; guard . retain (| (existing , _) | existing != & key_string) ; guard . push ((key_string , value)) ; } pub fn clone_store (& self) -> Vec < (String , Vec < u8 >) > { self . store . read () . clone () } pub fn from_entries (entries : Vec < (String , Vec < u8 >) >) -> Self { Self { store : Arc :: new (RwLock :: new (entries)) , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl Page { pub fn new (id : PageID , size : usize , location : PageLocation) -> Result < Self , PageError > { if size == 0 { return Err (PageError :: InvalidSize (size)) ; } let debug_id = PAGE_COUNTER . fetch_add (1 , Ordering :: Relaxed) ; println ! ("[PAGE {:>4}] NEW     id={:>6} size={:>7} loc={:?}" , debug_id , id . 0 , size , location) ; let (data_ptr , unified_cuda_backing) = if location == PageLocation :: Unified { # [cfg (feature = "cuda")] { let mut ptr : * mut c_void = ptr :: null_mut () ; let ret = unsafe { cudaMallocManaged (& mut ptr as * mut * mut c_void , size , 1) } ; if ret != 0 || ptr . is_null () { eprintln ! ("cudaMallocManaged failed (code {}), falling back to host allocation for unified page {}" , ret , id . 0) ; (allocate_zeroed (size , 1) ? , false) } else { (ptr as * mut u8 , true) } } # [cfg (not (feature = "cuda"))] { (allocate_zeroed (size , 1) ? , false) } } else { (allocate_zeroed (size , 1) ? , false) } ; let mask_size = (size + 7) / 8 ; let mask_ptr = allocate_zeroed (mask_size , 2) ? ; Ok (Self { debug_id , id , epoch : EpochCell :: new (0) , data : data_ptr , mask : mask_ptr , capacity : size , location , metadata : Metadata :: new () , unified_cuda_backing , }) } pub fn size (& self) -> usize { self . capacity } pub fn location (& self) -> PageLocation { self . location } pub fn data_slice (& self) -> & [u8] { unsafe { std :: slice :: from_raw_parts (self . data , self . capacity) } } pub fn data_mut_slice (& mut self) -> & mut [u8] { unsafe { std :: slice :: from_raw_parts_mut (self . data , self . capacity) } } pub fn mask_slice (& self) -> & [u8] { unsafe { std :: slice :: from_raw_parts (self . mask , (self . capacity + 7) / 8) } } pub fn data_ptr (& mut self) -> * mut u8 { self . data } pub fn mask_ptr (& mut self) -> * mut u8 { self . mask } pub fn epoch (& self) -> Epoch { self . epoch . load () } pub fn set_epoch (& self , epoch : Epoch) { self . epoch . store (epoch) ; } pub fn metadata_entries (& self) -> Vec < (String , Vec < u8 >) > { self . metadata . clone_store () } pub fn set_metadata (& mut self , entries : Vec < (String , Vec < u8 >) >) { self . metadata = Metadata :: from_entries (entries) ; } pub fn metadata_blob (& self) -> Vec < u8 > { let entries = self . metadata . clone_store () ; if entries . is_empty () { return Vec :: new () ; } let mut blob = Vec :: with_capacity (64) ; blob . extend_from_slice (& (entries . len () as u32) . to_le_bytes ()) ; for (key , value) in entries { let key_bytes = key . as_bytes () ; blob . extend_from_slice (& (key_bytes . len () as u32) . to_le_bytes ()) ; blob . extend_from_slice (key_bytes) ; blob . extend_from_slice (& (value . len () as u32) . to_le_bytes ()) ; blob . extend_from_slice (& value) ; } blob } pub fn set_metadata_blob (& mut self , blob : & [u8]) -> Result < () , PageError > { if blob . is_empty () { self . metadata = Metadata :: new () ; return Ok (()) ; } let mut cursor = 0usize ; let entry_count = read_u32 (blob , & mut cursor) ? as usize ; let mut entries = Vec :: with_capacity (entry_count) ; for _ in 0 .. entry_count { let key_len = read_u32 (blob , & mut cursor) ? as usize ; let key_bytes = read_bytes (blob , & mut cursor , key_len) ? ; let value_len = read_u32 (blob , & mut cursor) ? as usize ; let value_bytes = read_bytes (blob , & mut cursor , value_len) ? ; let key = String :: from_utf8 (key_bytes) . map_err (| _ | PageError :: MetadataDecode ("invalid utf-8 key")) ? ; entries . push ((key , value_bytes)) ; } self . metadata = Metadata :: from_entries (entries) ; Ok (()) } pub fn apply_delta (& mut self , delta : & Delta) -> Result < () , PageError > { if let Err (err) = delta_validation :: validate_delta (delta) { return Err (match err { DeltaError :: SizeMismatch { mask_len , payload_len } => PageError :: MaskSizeMismatch { expected : mask_len , found : payload_len , } , DeltaError :: PageIDMismatch { expected , found } => PageError :: PageIDMismatch { expected , found , } , DeltaError :: MaskSizeMismatch { expected , found } => PageError :: MaskSizeMismatch { expected , found , } , }) ; } if delta . page_id != self . id { return Err (PageError :: PageIDMismatch { expected : self . id , found : delta . page_id , }) ; } let mut payload_idx = 0usize ; for i in 0 .. self . capacity { let changed = if i < delta . mask . len () { delta . mask [i] } else { false } ; if changed { if delta . is_sparse { if payload_idx >= delta . payload . len () { return Err (PageError :: MaskSizeMismatch { expected : payload_idx , found : delta . payload . len () , }) ; } unsafe { * self . data . add (i) = delta . payload [payload_idx] ; } payload_idx += 1 ; } else { let payload_i = i . min (delta . payload . len () - 1) ; unsafe { * self . data . add (i) = delta . payload [payload_i] ; } } let mask_byte = unsafe { self . mask . add (i / 8) } ; unsafe { * mask_byte |= 1 << (i % 8) ; } } } self . epoch . store (delta . epoch) ; Ok (()) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl PageLocation { pub fn from_tag (tag : i32) -> Result < Self , PageError > { match tag { 0 => Ok (PageLocation :: Cpu) , 1 => Ok (PageLocation :: Gpu) , 2 => Ok (PageLocation :: Unified) , other => Err (PageError :: InvalidLocation (other)) , } } } . self_ty` (line 0, priv)
- [Rust | Function] `read_bytes` (line 0, priv)
  - Signature: `fn read_bytes (blob : & [u8] , cursor : & mut usize , len : usize) -> Result < Vec < u8 > , PageError > { if * cursor...`
  - Calls: len, Err, PageError::MetadataDecode, to_vec, Ok
- [Rust | Function] `read_u32` (line 0, priv)
  - Signature: `fn read_u32 (blob : & [u8] , cursor : & mut usize) -> Result < u32 , PageError > { if * cursor + 4 > blob . len () { ...`
  - Calls: len, Err, PageError::MetadataDecode, map_err, try_into, PageError::MetadataDecode, Ok, u32::from_le_bytes

## File: MMSB/src/01_page/simd_mask.rs

- Layer(s): 01_page
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `generate_mask` (line 0, pub)
  - Signature: `# [doc = " Generate a write mask for two equally sized byte slices."] pub fn generate_mask (old : & [u8] , new : & [u...`
  - Calls: collect, map, zip, iter, iter

## File: MMSB/src/01_page/tlog.rs

- Layer(s): 01_page
- Language coverage: Rust (10)
- Element types: Function (4), Impl (3), Struct (3)
- Total elements: 10

### Elements

- [Rust | Impl] `Drop for impl Drop for TransactionLogReader { fn drop (& mut self) { } } . self_ty` (line 0, priv)
- [Rust | Struct] `LogSummary` (line 0, pub)
  - Signature: `# [derive (Debug , Default , Clone , Copy)] pub struct LogSummary { pub total_deltas : u64 , pub total_bytes : u64 , ...`
- [Rust | Struct] `TransactionLog` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct TransactionLog { entries : RwLock < VecDeque < Delta > > , writer : RwLock < Option < B...`
- [Rust | Struct] `TransactionLogReader` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct TransactionLogReader { reader : BufReader < File > , version : u32 , }`
- [Rust | Impl] `impl TransactionLog { pub fn new (path : impl Into < PathBuf >) -> std :: io :: Result < Self > { let path = path . into () ; let mut file = OpenOptions :: new () . create (true) . append (true) . open (& path) ? ; if file . metadata () ? . len () == 0 { file . write_all (MAGIC) ? ; file . write_all (& VERSION . to_le_bytes ()) ? ; file . flush () ? ; } let writer = BufWriter :: new (file) ; Ok (Self { entries : RwLock :: new (VecDeque :: new ()) , writer : RwLock :: new (Some (writer)) , path , }) } pub fn append (& self , delta : Delta) -> std :: io :: Result < () > { { self . entries . write () . push_back (delta . clone ()) ; } if let Some (writer) = self . writer . write () . as_mut () { serialize_frame (writer , & delta) ? ; writer . flush () ? ; Ok (()) } else { Err (std :: io :: Error :: new (std :: io :: ErrorKind :: Other , "transaction log writer closed" ,)) } } pub fn len (& self) -> usize { self . entries . read () . len () } pub fn drain (& self) -> Vec < Delta > { let mut guard = self . entries . write () ; guard . drain (..) . collect () } pub fn current_offset (& self) -> std :: io :: Result < u64 > { let writer_lock = self . writer . read () ; if let Some (writer) = writer_lock . as_ref () { writer . get_ref () . metadata () . map (| meta | meta . len ()) } else { Err (std :: io :: Error :: new (std :: io :: ErrorKind :: Other , "transaction log writer unavailable" ,)) } } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl TransactionLogReader { pub fn open (path : impl AsRef < Path >) -> std :: io :: Result < Self > { let file = File :: open (path) ? ; let mut reader = BufReader :: new (file) ; let version = validate_header (& mut reader) ? ; Ok (Self { reader , version }) } pub fn next (& mut self) -> std :: io :: Result < Option < Delta > > { read_frame (& mut self . reader , self . version) } pub fn free (self) { } } . self_ty` (line 0, priv)
- [Rust | Function] `read_frame` (line 0, priv)
  - Signature: `fn read_frame (reader : & mut BufReader < File > , version : u32) -> std :: io :: Result < Option < Delta > > { let m...`
  - Calls: read_exact, kind, Ok, Err, read_exact, read_exact, read_exact, u32::from_le_bytes, read_exact, collect, map, iter, read_exact, u32::from_le_bytes, read_exact, read_exact, read_exact, read_exact, u32::from_le_bytes, read_exact, Source, to_string, String::from_utf8_lossy, is_err, read_exact, Ok, u32::from_le_bytes, read_exact, Some, to_string, String::from_utf8_lossy, Ok, Some, DeltaID, u64::from_le_bytes, PageID, u64::from_le_bytes, Epoch, u32::from_le_bytes, u64::from_le_bytes
- [Rust | Function] `serialize_frame` (line 0, priv)
  - Signature: `fn serialize_frame (writer : & mut BufWriter < File > , delta : & Delta) -> std :: io :: Result < () > { writer . wri...`
  - Calls: write_all, to_le_bytes, write_all, to_le_bytes, write_all, to_le_bytes, len, write_all, to_le_bytes, write_all, len, write_all, to_le_bytes, write_all, write_all, write_all, to_le_bytes, as_bytes, write_all, to_le_bytes, len, write_all, unwrap_or, map, as_ref, len, as_bytes, write_all, to_le_bytes, write_all, as_bytes, Ok
- [Rust | Function] `summary` (line 0, pub)
  - Signature: `pub fn summary (path : impl AsRef < Path >) -> std :: io :: Result < LogSummary > { let file = match File :: open (pa...`
  - Calls: File::open, as_ref, kind, Err, Err, len, metadata, Ok, LogSummary::default, BufReader::new, validate_header, LogSummary::default, read_frame, unwrap_or, map, as_ref, len, as_bytes, len, len, max, Ok
- [Rust | Function] `validate_header` (line 0, priv)
  - Signature: `fn validate_header (reader : & mut BufReader < File >) -> std :: io :: Result < u32 > { reader . seek (SeekFrom :: St...`
  - Calls: seek, SeekFrom::Start, read_exact, Err, std::io::Error::new, read_exact, u32::from_le_bytes, Err, std::io::Error::new, Ok

## File: MMSB/src/01_page/tlog_compression.rs

- Layer(s): 01_page
- Language coverage: Rust (8)
- Element types: Enum (1), Function (6), Struct (1)
- Total elements: 8

### Elements

- [Rust | Enum] `CompressionMode` (line 0, pub)
- [Rust | Struct] `CompressionStats` (line 0, pub)
  - Signature: `pub struct CompressionStats { pub original_size : usize , pub compressed_size : usize , pub ratio : f64 , }`
- [Rust | Function] `bitpack_mask` (line 0, priv)
  - Signature: `# [doc = " Bitpack dense boolean masks"] fn bitpack_mask (mask : & [bool]) -> Vec < u8 > { let num_bytes = (mask . le...`
  - Calls: len, enumerate, iter
- [Rust | Function] `bitunpack_mask` (line 0, priv)
  - Signature: `fn bitunpack_mask (packed : & [u8] , output : & mut [bool]) { for (i , out) in output . iter_mut () . enumerate () { ...`
  - Calls: enumerate, iter_mut, len
- [Rust | Function] `compact` (line 0, pub)
  - Signature: `pub fn compact (deltas : & [Delta]) -> Vec < Delta > { if deltas . len () <= 1 { return deltas . to_vec () ; } let mu...`
  - Calls: len, to_vec, Vec::with_capacity, len, iter, next, push, clone, last_mut, merge, push, clone
- [Rust | Function] `compress_delta_mask` (line 0, pub)
  - Signature: `pub fn compress_delta_mask (mask : & [bool] , mode : CompressionMode) -> (Vec < u8 > , CompressionStats) { let origin...`
  - Calls: len, collect, map, iter, encode_rle, bitpack_mask, len, max, len
- [Rust | Function] `decode_rle` (line 0, priv)
  - Signature: `fn decode_rle (encoded : & [u8] , output : & mut [bool]) { let mut pos = 0 ; for & byte in encoded { let is_zero = (b...`
  - Calls: len
- [Rust | Function] `encode_rle` (line 0, priv)
  - Signature: `# [doc = " RLE (Run-Length Encoding) for sparse masks"] fn encode_rle (mask : & [bool]) -> Vec < u8 > { let mut encod...`
  - Calls: Vec::new, is_empty, push, push

## File: MMSB/src/01_page/tlog_replay.rs

- Layer(s): 01_page
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `apply_log` (line 0, pub)
  - Signature: `pub fn apply_log (pages : & mut [Page] , deltas : & [Delta]) { for delta in deltas { if let Some (page) = pages . ite...`
  - Calls: find, iter_mut, apply_delta

## File: MMSB/src/01_page/tlog_serialization.rs

- Layer(s): 01_page
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `read_log` (line 0, pub)
  - Signature: `pub fn read_log (path : impl AsRef < Path >) -> std :: io :: Result < Vec < Delta > > { let file = File :: open (path...`
  - Calls: File::open, BufReader::new, read_exact, Err, std::io::Error::new, read_exact, u32::from_le_bytes, Vec::new, is_err, read_exact, read_exact, read_exact, read_exact, u32::from_le_bytes, read_exact, collect, map, iter, read_exact, u32::from_le_bytes, read_exact, read_exact, read_exact, u64::from_le_bytes, read_exact, u32::from_le_bytes, read_exact, Source, to_string, String::from_utf8_lossy, is_err, read_exact, u32::from_le_bytes, read_exact, Some, to_string, String::from_utf8_lossy, push, DeltaID, u64::from_le_bytes, PageID, u64::from_le_bytes, Epoch, u32::from_le_bytes, Ok

