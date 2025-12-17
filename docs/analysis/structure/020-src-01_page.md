# Structure Group: src/01_page

## File: MMSB/src/01_page/allocator.rs

- Layer(s): 01_page
- Language coverage: Rust (10)
- Element types: Function (3), Impl (2), Module (1), Struct (4)
- Total elements: 10

### Elements

- [Rust | Impl] `Default for impl Default for PageAllocatorConfig { fn default () -> Self { Self { default_location : PageLocation :: Cpu , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `PageAllocator` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct PageAllocator { config : PageAllocatorConfig , pages : Mutex < HashMap < PageID , Box <...`
- [Rust | Struct] `PageAllocatorConfig` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageAllocatorConfig { pub default_location : PageLocation , }`
- [Rust | Struct] `PageInfo` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageInfo { pub page_id : PageID , pub size : usize , pub location : PageLocatio...`
- [Rust | Struct] `PageSnapshotData` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PageSnapshotData { pub page_id : PageID , pub size : usize , pub location : Pag...`
- [Rust | Impl] `impl PageAllocator { pub fn new (config : PageAllocatorConfig) -> Self { Self :: with_stats (config , Arc :: new (AllocatorStats :: default ())) } pub fn with_stats (config : PageAllocatorConfig , stats : Arc < AllocatorStats >) -> Self { println ! ("Allocating new PageAllocator instance with config: {:?}" , config) ; Self { config , pages : Mutex :: new (HashMap :: new ()) , next_id : AtomicU64 :: new (1) , stats , } } pub fn allocate_raw (& self , page_id_hint : PageID , size : usize , location : Option < PageLocation >) -> Result < * mut Page , PageError > { let loc = location . unwrap_or (self . config . default_location) ; if self . pages . lock () . contains_key (& page_id_hint) { return Err (PageError :: AlreadyExists (page_id_hint)) ; } let page = Box :: new (Page :: new (page_id_hint , size , loc) ?) ; let ptr = Box :: into_raw (page) ; println ! ("[ALLOCATOR] allocate_raw(id={}) → raw ptr = {:p}" , page_id_hint . 0 , ptr) ; self . pages . lock () . insert (page_id_hint , unsafe { Box :: from_raw (ptr) }) ; self . stats . record_alloc () ; Ok (ptr) } pub fn free (& self , page_id : PageID) { if let Some (_) = self . pages . lock () . remove (& page_id) { println ! ("[ALLOCATOR] Freed page {}" , page_id . 0) ; self . stats . record_free () ; } } pub fn release (& self , page_id : PageID) { if let Some (boxed_page) = self . pages . lock () . remove (& page_id) { println ! ("[ALLOCATOR] release({}): ownership transferred — Box removed from map but NOT dropped (caller now owns it)" , page_id . 0) ; std :: mem :: forget (boxed_page) ; } else { println ! ("[ALLOCATOR] release({}): page not found — already released?" , page_id . 0) ; } } pub fn acquire_page (& self , page_id : PageID) -> Option < * mut Page > { self . pages . lock () . get (& page_id) . map (| b | & * * b as * const Page as * mut Page) } pub fn len (& self) -> usize { self . pages . lock () . len () } pub fn page_infos (& self) -> Vec < PageInfo > { let pages = self . pages . lock () ; pages . values () . map (| page | PageInfo { page_id : page . id , size : page . size () , location : page . location () , epoch : page . epoch () . 0 , metadata : page . metadata_blob () , }) . collect () } pub fn snapshot_pages (& self) -> Vec < PageSnapshotData > { let pages = self . pages . lock () ; pages . values () . map (| page | PageSnapshotData { page_id : page . id , size : page . size () , location : page . location () , epoch : page . epoch () . 0 , metadata_blob : page . metadata_blob () , data : page . data_slice () . to_vec () , }) . collect () } pub fn restore_from_snapshot (& self , snapshots : Vec < PageSnapshotData >) -> Result < () , PageError > { eprintln ! ("\n=== RESTORE_FROM_SNAPSHOT STARTED ===") ; eprintln ! ("   Clearing {} existing pages" , self . pages . lock () . len ()) ; let mut pages = self . pages . lock () ; pages . clear () ; for (i , snapshot) in snapshots . iter () . enumerate () { eprintln ! ("   [{i}] Restoring page ID={:?} size={} epoch={} loc={:?}" , snapshot . page_id , snapshot . size , snapshot . epoch , snapshot . location) ; let mut page = match Page :: new (snapshot . page_id , snapshot . size , snapshot . location) { Ok (p) => Box :: new (p) , Err (e) => { eprintln ! ("      Page::new() FAILED: {e}") ; return Err (e) ; } } ; page . set_epoch (Epoch (snapshot . epoch)) ; eprintln ! ("      Epoch set to {}" , snapshot . epoch) ; let dst = page . data_mut_slice () ; if dst . len () != snapshot . data . len () { eprintln ! ("      FATAL: data size mismatch! page={} snapshot={}" , dst . len () , snapshot . data . len ()) ; return Err (PageError :: MetadataDecode ("data size mismatch in snapshot")) ; } dst . copy_from_slice (& snapshot . data) ; eprintln ! ("      Data copied ({} bytes)" , snapshot . data . len ()) ; eprintln ! ("      Applying metadata ({} bytes)..." , snapshot . metadata_blob . len ()) ; if let Err (e) = page . set_metadata_blob (& snapshot . metadata_blob) { eprintln ! ("      METADATA RESTORE FAILED: {e}") ; return Err (e) ; } eprintln ! ("      Metadata restored OK") ; pages . insert (snapshot . page_id , page) ; eprintln ! ("      Page inserted") ; } eprintln ! ("=== RESTORE_FROM_SNAPSHOT SUCCESS: {} pages restored ===" , snapshots . len ()) ; Ok (()) } pub fn stats (& self) -> Arc < AllocatorStats > { Arc :: clone (& self . stats) } } . self_ty` (line 0, priv)
- [Rust | Function] `test_checkpoint_roundtrip_in_memory` (line 0, priv)
  - Signature: `# [test] fn test_checkpoint_roundtrip_in_memory () { let alloc = PageAllocator :: new (PageAllocatorConfig :: default...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, apply_delta, DeltaID, PageID, Epoch, Source, into, snapshot_pages, expect, restore_from_snapshot, unwrap, acquire_page, PageID
- [Rust | Function] `test_page_info_metadata_roundtrip` (line 0, priv)
  - Signature: `# [test] fn test_page_info_metadata_roundtrip () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defau...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, set_metadata, page_infos
- [Rust | Function] `test_unified_page` (line 0, priv)
  - Signature: `# [test] fn test_unified_page () { let config = PageAllocatorConfig { default_location : PageLocation :: Unified , } ...`
  - Calls: PageAllocator::new, expect, allocate_raw, PageID, data_mut_slice
- [Rust | Module] `tests` (line 0, priv)

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

## File: MMSB/src/01_page/columnar_delta.rs

- Layer(s): 01_page
- Language coverage: Rust (9)
- Element types: Function (4), Impl (3), Module (1), Struct (1)
- Total elements: 9

### Elements

- [Rust | Struct] `ColumnarDeltaBatch` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct ColumnarDeltaBatch { count : usize , delta_ids : Vec < DeltaID > , page_ids : V...`
- [Rust | Impl] `Default for impl Default for ColumnarDeltaBatch { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Impl] `From for impl From < Vec < Delta > > for ColumnarDeltaBatch { fn from (value : Vec < Delta >) -> Self { Self :: from_rows (value) } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl ColumnarDeltaBatch { pub fn new () -> Self { Self { count : 0 , delta_ids : Vec :: new () , page_ids : Vec :: new () , epochs : Vec :: new () , is_sparse : Vec :: new () , mask_offsets : Vec :: new () , payload_offsets : Vec :: new () , mask_pool : Vec :: new () , payload_pool : Vec :: new () , timestamps : Vec :: new () , sources : Vec :: new () , metadata : Vec :: new () , } } pub fn from_rows (deltas : Vec < Delta >) -> Self { let mut batch = Self :: new () ; batch . extend (deltas) ; batch } pub fn len (& self) -> usize { self . count } pub fn is_empty (& self) -> bool { self . count == 0 } pub fn iter (& self) -> impl Iterator < Item = Delta > + '_ { (0 .. self . count) . map (move | idx | self . delta_at (idx) . expect ("index in range")) } pub fn extend < I > (& mut self , deltas : I) where I : IntoIterator < Item = Delta > , { for delta in deltas { let mask_start = self . mask_pool . len () ; self . mask_pool . extend (delta . mask . iter () . map (| flag | if * flag { 1 } else { 0 })) ; self . mask_offsets . push ((mask_start , delta . mask . len ())) ; let payload_start = self . payload_pool . len () ; self . payload_pool . extend (& delta . payload) ; self . payload_offsets . push ((payload_start , delta . payload . len ())) ; self . delta_ids . push (delta . delta_id) ; self . page_ids . push (delta . page_id) ; self . epochs . push (delta . epoch) ; self . is_sparse . push (delta . is_sparse) ; self . timestamps . push (delta . timestamp) ; self . sources . push (delta . source) ; self . metadata . push (delta . intent_metadata) ; self . count += 1 ; } } pub fn to_vec (& self) -> Vec < Delta > { self . iter () . collect () } pub fn filter_epoch_eq (& self , epoch : Epoch) -> Vec < usize > { let mut matches = Vec :: new () ; let words = self . epoch_words () ; let mut idx = 0usize ; const LANES : usize = 8 ; while idx + LANES <= words . len () { let mut lane_mask = 0u8 ; for lane in 0 .. LANES { if words [idx + lane] == epoch . 0 { lane_mask |= 1 << lane ; } } if lane_mask != 0 { for lane in 0 .. LANES { if (lane_mask >> lane) & 1 == 1 { matches . push (idx + lane) ; } } } idx += LANES ; } for (offset , value) in words [idx ..] . iter () . enumerate () { if * value == epoch . 0 { matches . push (idx + offset) ; } } matches } pub fn scan_page_id (& self , target : PageID) -> Vec < usize > { let mut matches = Vec :: new () ; let ids = self . page_words () ; let mut idx = 0usize ; const LANES : usize = 4 ; while idx + LANES <= ids . len () { let mut lane_mask = 0u8 ; for lane in 0 .. LANES { if ids [idx + lane] == target . 0 { lane_mask |= 1 << lane ; } } if lane_mask != 0 { for lane in 0 .. LANES { if (lane_mask >> lane) & 1 == 1 { matches . push (idx + lane) ; } } } idx += LANES ; } for (offset , value) in ids [idx ..] . iter () . enumerate () { if * value == target . 0 { matches . push (idx + offset) ; } } matches } pub fn apply_to_pages (& self , pages : & mut HashMap < PageID , Page > ,) -> Result < () , PageError > { for idx in 0 .. self . count { let delta = self . delta_at (idx) . expect ("index valid") ; if let Some (page) = pages . get_mut (& delta . page_id) { page . apply_delta (& delta) ? ; } else { return Err (PageError :: PageNotFound (delta . page_id)) ; } } Ok (()) } pub fn delta_at (& self , idx : usize) -> Option < Delta > { (idx < self . count) . then (| | self . build_delta (idx)) } pub fn page_id_at (& self , idx : usize) -> Option < PageID > { self . page_ids . get (idx) . copied () } fn build_delta (& self , idx : usize) -> Delta { let (mask_start , mask_len) = self . mask_offsets [idx] ; let mask = self . mask_pool [mask_start .. mask_start + mask_len] . iter () . map (| b | * b != 0) . collect :: < Vec < bool > > () ; let (payload_start , payload_len) = self . payload_offsets [idx] ; let payload = self . payload_pool [payload_start .. payload_start + payload_len] . to_vec () ; Delta { delta_id : self . delta_ids [idx] , page_id : self . page_ids [idx] , epoch : self . epochs [idx] , mask , payload , is_sparse : self . is_sparse [idx] , timestamp : self . timestamps [idx] , source : self . sources [idx] . clone () , intent_metadata : self . metadata [idx] . clone () , } } fn epoch_words (& self) -> & [u32] { unsafe { std :: slice :: from_raw_parts (self . epochs . as_ptr () as * const u32 , self . epochs . len ()) } } fn page_words (& self) -> & [u64] { unsafe { std :: slice :: from_raw_parts (self . page_ids . as_ptr () as * const u64 , self . page_ids . len ()) } } } . self_ty` (line 0, priv)
- [Rust | Function] `make_delta` (line 0, priv)
  - Signature: `fn make_delta (id : u64 , page : u64 , epoch : u32 , payload : & [u8]) -> Delta { Delta { delta_id : DeltaID (id) , p...`
  - Calls: DeltaID, PageID, Epoch, collect, map, iter, to_vec, Source
- [Rust | Function] `test_apply_to_pages` (line 0, priv)
  - Signature: `# [test] fn test_apply_to_pages () { let deltas = vec ! [make_delta (1 , 1 , 1 , b"\x01\x02") , make_delta (2 , 2 , 2...`
  - Calls: ColumnarDeltaBatch::from_rows, HashMap::new, unwrap, Page::new, PageID, unwrap, Page::new, PageID, insert, PageID, insert, PageID, unwrap, apply_to_pages, unwrap, remove, PageID, unwrap, remove, PageID
- [Rust | Function] `test_epoch_filter` (line 0, priv)
  - Signature: `# [test] fn test_epoch_filter () { let deltas = vec ! [make_delta (1 , 1 , 1 , b"a") , make_delta (2 , 2 , 2 , b"b") ...`
  - Calls: ColumnarDeltaBatch::from_rows, filter_epoch_eq, Epoch
- [Rust | Function] `test_roundtrip` (line 0, priv)
  - Signature: `# [test] fn test_roundtrip () { let deltas = vec ! [make_delta (1 , 10 , 5 , b"abc") , make_delta (2 , 10 , 6 , b"def...`
  - Calls: ColumnarDeltaBatch::from_rows, clone, to_vec
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/01_page/delta.rs

- Layer(s): 01_page
- Language coverage: Rust (3)
- Element types: Function (1), Impl (1), Struct (1)
- Total elements: 3

### Elements

- [Rust | Struct] `Delta` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct Delta { pub delta_id : DeltaID , pub page_id : PageID , pub epoch : Epoch , pub...`
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

## File: MMSB/src/01_page/device.rs

- Layer(s): 01_page
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `DeviceRegistry` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct DeviceRegistry { pages : RwLock < HashMap < PageID , Arc < Page > > > , }`
- [Rust | Impl] `impl DeviceRegistry { pub fn register (& self , page : Arc < Page >) { self . pages . write () . insert (page . id , page) ; } pub fn unregister (& self , page_id : PageID) { self . pages . write () . remove (& page_id) ; } pub fn get (& self , page_id : PageID) -> Option < Arc < Page > > { self . pages . read () . get (& page_id) . cloned () } } . self_ty` (line 0, priv)

## File: MMSB/src/01_page/device_registry.rs

- Layer(s): 01_page
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `DeviceBufferRegistry` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct DeviceBufferRegistry { map : RwLock < HashMap < PageID , Arc < Page > > > , }`
- [Rust | Impl] `impl DeviceBufferRegistry { pub fn insert (& self , page : Arc < Page >) { self . map . write () . insert (page . id , page) ; } pub fn remove (& self , page_id : PageID) { self . map . write () . remove (& page_id) ; } pub fn len (& self) -> usize { self . map . read () . len () } pub fn contains (& self , page_id : PageID) -> bool { self . map . read () . contains_key (& page_id) } pub fn get (& self , page_id : PageID) -> Option < Arc < Page > > { self . map . read () . get (& page_id) . cloned () } } . self_ty` (line 0, priv)

## File: MMSB/src/01_page/host_device_sync.rs

- Layer(s): 01_page
- Language coverage: Rust (2)
- Element types: Impl (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `HostDeviceSync` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct HostDeviceSync { pending : Vec < PageID > , }`
- [Rust | Impl] `impl HostDeviceSync { pub fn enqueue (& mut self , page_id : PageID) { self . pending . push (page_id) ; } pub fn drain (& mut self) -> Vec < PageID > { std :: mem :: take (& mut self . pending) } } . self_ty` (line 0, priv)

## File: MMSB/src/01_page/integrity_checker.rs

- Layer(s): 01_page
- Language coverage: Rust (11)
- Element types: Enum (1), Function (4), Impl (2), Module (1), Struct (3)
- Total elements: 11

### Elements

- [Rust | Struct] `DeltaIntegrityChecker` (line 0, pub)
  - Signature: `pub struct DeltaIntegrityChecker { registry : Arc < DeviceBufferRegistry > , last_epoch : HashMap < PageID , Epoch > , }`
- [Rust | Struct] `IntegrityReport` (line 0, pub)
  - Signature: `# [derive (Debug , Default)] pub struct IntegrityReport { pub total : usize , pub violations : Vec < IntegrityViolati...`
- [Rust | Struct] `IntegrityViolation` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct IntegrityViolation { pub delta_id : DeltaID , pub page_id : PageID , pub kind :...`
- [Rust | Enum] `IntegrityViolationKind` (line 0, pub)
- [Rust | Function] `delta` (line 0, priv)
  - Signature: `fn delta (delta_id : u64 , page_id : u64 , epoch : u32 , payload : & [u8]) -> Delta { Delta { delta_id : DeltaID (del...`
  - Calls: DeltaID, PageID, Epoch, collect, map, iter, to_vec, Source, into
- [Rust | Function] `detects_orphan_and_epoch_errors` (line 0, priv)
  - Signature: `# [test] fn detects_orphan_and_epoch_errors () { let registry = Arc :: new (DeviceBufferRegistry :: default ()) ; reg...`
  - Calls: Arc::new, DeviceBufferRegistry::default, insert, page, DeltaIntegrityChecker::new, Arc::clone, validate
- [Rust | Impl] `impl DeltaIntegrityChecker { pub fn new (registry : Arc < DeviceBufferRegistry >) -> Self { Self { registry , last_epoch : HashMap :: new () , } } pub fn validate (& mut self , deltas : & [Delta]) -> IntegrityReport { let mut report = IntegrityReport { total : deltas . len () , violations : Vec :: new () , } ; for delta in deltas { if ! self . registry . contains (delta . page_id) { report . violations . push (IntegrityViolation { delta_id : delta . delta_id , page_id : delta . page_id , kind : IntegrityViolationKind :: OrphanDelta , }) ; continue ; } if ! schema_valid (delta) { report . violations . push (IntegrityViolation { delta_id : delta . delta_id , page_id : delta . page_id , kind : IntegrityViolationKind :: SchemaMismatch { mask_len : delta . mask . len () , payload_len : delta . payload . len () , } , }) ; continue ; } let last_epoch = self . last_epoch . entry (delta . page_id) . or_insert (Epoch (0)) ; if delta . epoch . 0 < last_epoch . 0 { report . violations . push (IntegrityViolation { delta_id : delta . delta_id , page_id : delta . page_id , kind : IntegrityViolationKind :: EpochRegression { previous : * last_epoch , current : delta . epoch , } , }) ; } else { * last_epoch = delta . epoch ; } } report } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl IntegrityReport { pub fn passed (& self) -> bool { self . violations . is_empty () } } . self_ty` (line 0, priv)
- [Rust | Function] `page` (line 0, priv)
  - Signature: `fn page (page_id : u64) -> Arc < Page > { Arc :: new (Page :: new (PageID (page_id) , 4 , PageLocation :: Cpu) . unwr...`
  - Calls: Arc::new, unwrap, Page::new, PageID
- [Rust | Function] `schema_valid` (line 0, priv)
  - Signature: `fn schema_valid (delta : & Delta) -> bool { if delta . is_sparse { let changed = delta . mask . iter () . filter (| f...`
  - Calls: count, filter, iter, len, len, len
- [Rust | Module] `tests` (line 0, priv)

## File: MMSB/src/01_page/lockfree_allocator.rs

- Layer(s): 01_page
- Language coverage: Rust (6)
- Element types: Impl (4), Struct (2)
- Total elements: 6

### Elements

- [Rust | Impl] `Drop for impl Drop for LockFreeAllocator { fn drop (& mut self) { self . clear () ; } } . self_ty` (line 0, priv)
- [Rust | Struct] `FreeListNode` (line 0, priv)
  - Signature: `struct FreeListNode { next : AtomicPtr < FreeListNode > , page_ptr : * mut Page , }`
- [Rust | Struct] `LockFreeAllocator` (line 0, pub)
  - Signature: `pub struct LockFreeAllocator { freelist_head : AtomicPtr < FreeListNode > , freelist_size : AtomicU64 , allocated_cou...`
- [Rust | Impl] `Send for unsafe impl Send for LockFreeAllocator { } . self_ty` (line 0, priv)
- [Rust | Impl] `Sync for unsafe impl Sync for LockFreeAllocator { } . self_ty` (line 0, priv)
- [Rust | Impl] `impl LockFreeAllocator { pub fn new () -> Self { Self { freelist_head : AtomicPtr :: new (ptr :: null_mut ()) , freelist_size : AtomicU64 :: new (0) , allocated_count : AtomicU64 :: new (0) , freed_count : AtomicU64 :: new (0) , } } pub fn try_allocate_small (& self , _page_id : PageID , size : usize , _location : PageLocation) -> Option < * mut Page > { if size > SMALL_PAGE_THRESHOLD { return None ; } loop { let head = self . freelist_head . load (Ordering :: Acquire) ; if head . is_null () { return None ; } let node = unsafe { & * head } ; let next = node . next . load (Ordering :: Relaxed) ; if self . freelist_head . compare_exchange (head , next , Ordering :: Release , Ordering :: Acquire) . is_ok () { self . freelist_size . fetch_sub (1 , Ordering :: Relaxed) ; self . allocated_count . fetch_add (1 , Ordering :: Relaxed) ; let page_ptr = node . page_ptr ; unsafe { let _ = Box :: from_raw (head) ; } return Some (page_ptr) ; } } } pub fn deallocate_small (& self , page_ptr : * mut Page) -> bool { let page = unsafe { & * page_ptr } ; if page . size () > SMALL_PAGE_THRESHOLD { return false ; } let current_size = self . freelist_size . load (Ordering :: Relaxed) ; if current_size >= FREELIST_CAPACITY as u64 { return false ; } let node = Box :: into_raw (Box :: new (FreeListNode { next : AtomicPtr :: new (ptr :: null_mut ()) , page_ptr , })) ; loop { let head = self . freelist_head . load (Ordering :: Acquire) ; unsafe { (* node) . next . store (head , Ordering :: Relaxed) ; } if self . freelist_head . compare_exchange (head , node , Ordering :: Release , Ordering :: Acquire) . is_ok () { self . freelist_size . fetch_add (1 , Ordering :: Relaxed) ; self . freed_count . fetch_add (1 , Ordering :: Relaxed) ; return true ; } } } pub fn get_stats (& self) -> (u64 , u64 , u64) { (self . freelist_size . load (Ordering :: Relaxed) , self . allocated_count . load (Ordering :: Relaxed) , self . freed_count . load (Ordering :: Relaxed) ,) } pub fn clear (& self) { let mut head = self . freelist_head . swap (ptr :: null_mut () , Ordering :: AcqRel) ; while ! head . is_null () { let node = unsafe { Box :: from_raw (head) } ; head = node . next . load (Ordering :: Relaxed) ; unsafe { drop (Box :: from_raw (node . page_ptr)) ; } } self . freelist_size . store (0 , Ordering :: Relaxed) ; } } . self_ty` (line 0, priv)

## File: MMSB/src/01_page/mod.rs

- Layer(s): 01_page
- Language coverage: Rust (19)
- Element types: Module (19)
- Total elements: 19

### Elements

- [Rust | Module] `allocator` (line 0, pub)
- [Rust | Module] `checkpoint` (line 0, pub)
- [Rust | Module] `columnar_delta` (line 0, pub)
- [Rust | Module] `delta` (line 0, pub)
- [Rust | Module] `delta_merge` (line 0, pub)
- [Rust | Module] `delta_validation` (line 0, pub)
- [Rust | Module] `device` (line 0, pub)
- [Rust | Module] `device_registry` (line 0, pub)
- [Rust | Module] `epoch` (line 0, pub)
- [Rust | Module] `host_device_sync` (line 0, pub)
- [Rust | Module] `integrity_checker` (line 0, pub)
- [Rust | Module] `lockfree_allocator` (line 0, pub)
- [Rust | Module] `page` (line 0, pub)
- [Rust | Module] `replay_validator` (line 0, pub)
- [Rust | Module] `simd_mask` (line 0, pub)
- [Rust | Module] `tlog` (line 0, pub)
- [Rust | Module] `tlog_compression` (line 0, pub)
- [Rust | Module] `tlog_replay` (line 0, pub)
- [Rust | Module] `tlog_serialization` (line 0, pub)

## File: MMSB/src/01_page/page.rs

- Layer(s): 01_page
- Language coverage: Rust (11)
- Element types: Function (3), Impl (6), Struct (2)
- Total elements: 11

### Elements

- [Rust | Impl] `Clone for impl Clone for Page { fn clone (& self) -> Self { let new_debug_id = PAGE_COUNTER . fetch_add (1 , Ordering :: Relaxed) ; println ! ("[PAGE {:>4}] CLONE → [PAGE {:>4}]  id={:>6}" , self . debug_id , new_debug_id , self . id . 0) ; let layout_data = std :: alloc :: Layout :: array :: < u8 > (self . capacity) . expect ("invalid capacity") ; let data = unsafe { std :: alloc :: alloc (layout_data) } ; unsafe { std :: ptr :: copy_nonoverlapping (self . data , data , self . capacity) ; } let mask_size = (self . capacity + 7) / 8 ; let layout_mask = std :: alloc :: Layout :: array :: < u8 > (mask_size) . expect ("invalid mask size") ; let mask = unsafe { std :: alloc :: alloc (layout_mask) } ; unsafe { std :: ptr :: copy_nonoverlapping (self . mask , mask , mask_size) ; } Self { debug_id : new_debug_id , id : self . id , epoch : EpochCell :: new (self . epoch . load () . 0) , data , mask , capacity : self . capacity , location : self . location , metadata : self . metadata . clone () , unified_cuda_backing : false , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `Drop for impl Drop for Page { fn drop (& mut self) { println ! ("[PAGE {:>4}] DROP    id={:>6} loc={:?}" , self . debug_id , self . id . 0 , self . location) ; let mask_size = (self . capacity + 7) / 8 ; let mask_layout = std :: alloc :: Layout :: array :: < u8 > (mask_size) . unwrap () ; unsafe { std :: alloc :: dealloc (self . mask , mask_layout) } ; if self . location == PageLocation :: Unified && self . unified_cuda_backing { # [cfg (feature = "cuda")] unsafe { let _ = cudaFree (self . data as * mut c_void) ; } } else { unsafe { let layout = std :: alloc :: Layout :: array :: < u8 > (self . capacity) . unwrap () ; std :: alloc :: dealloc (self . data , layout) ; } } } } . self_ty` (line 0, priv)
- [Rust | Struct] `Metadata` (line 0, pub)
  - Signature: `# [doc = " Metadata key-value store with copy-on-write semantics."] # [derive (Debug , Clone , Default)] pub struct M...`
- [Rust | Struct] `Page` (line 0, pub)
  - Signature: `# [doc = " Memory page implementation shared across the runtime layers."] # [derive (Debug)] pub struct Page { debug_...`
- [Rust | Impl] `Send for unsafe impl Send for Page { } . self_ty` (line 0, priv)
- [Rust | Impl] `Sync for unsafe impl Sync for Page { } . self_ty` (line 0, priv)
- [Rust | Function] `allocate_zeroed` (line 0, priv)
  - Signature: `fn allocate_zeroed (size : usize , err_code : i32) -> Result < * mut u8 , PageError > { let layout = std :: alloc :: ...`
  - Calls: map_err, std::alloc::Layout::array, PageError::AllocError, std::alloc::alloc_zeroed, is_null, Err, PageError::AllocError, Ok
- [Rust | Impl] `impl Metadata { pub fn new () -> Self { Self { store : Arc :: new (RwLock :: new (Vec :: new ())) , } } pub fn insert (& self , key : impl Into < String > , value : Vec < u8 >) { let key_string = key . into () ; let mut guard = self . store . write () ; guard . retain (| (existing , _) | existing != & key_string) ; guard . push ((key_string , value)) ; } pub fn clone_store (& self) -> Vec < (String , Vec < u8 >) > { self . store . read () . clone () } pub fn from_entries (entries : Vec < (String , Vec < u8 >) >) -> Self { Self { store : Arc :: new (RwLock :: new (entries)) , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl Page { pub fn new (id : PageID , size : usize , location : PageLocation) -> Result < Self , PageError > { if size == 0 { return Err (PageError :: InvalidSize (size)) ; } let debug_id = PAGE_COUNTER . fetch_add (1 , Ordering :: Relaxed) ; println ! ("[PAGE {:>4}] NEW     id={:>6} size={:>7} loc={:?}" , debug_id , id . 0 , size , location) ; let (data_ptr , unified_cuda_backing) = if location == PageLocation :: Unified { # [cfg (feature = "cuda")] { let mut ptr : * mut c_void = ptr :: null_mut () ; let ret = unsafe { cudaMallocManaged (& mut ptr as * mut * mut c_void , size , 1) } ; if ret != 0 || ptr . is_null () { eprintln ! ("cudaMallocManaged failed (code {}), falling back to host allocation for unified page {}" , ret , id . 0) ; (allocate_zeroed (size , 1) ? , false) } else { (ptr as * mut u8 , true) } } # [cfg (not (feature = "cuda"))] { (allocate_zeroed (size , 1) ? , false) } } else { (allocate_zeroed (size , 1) ? , false) } ; let mask_size = (size + 7) / 8 ; let mask_ptr = allocate_zeroed (mask_size , 2) ? ; Ok (Self { debug_id , id , epoch : EpochCell :: new (0) , data : data_ptr , mask : mask_ptr , capacity : size , location , metadata : Metadata :: new () , unified_cuda_backing , }) } pub fn size (& self) -> usize { self . capacity } pub fn location (& self) -> PageLocation { self . location } pub fn data_slice (& self) -> & [u8] { unsafe { std :: slice :: from_raw_parts (self . data , self . capacity) } } pub fn data_mut_slice (& mut self) -> & mut [u8] { unsafe { std :: slice :: from_raw_parts_mut (self . data , self . capacity) } } pub fn mask_slice (& self) -> & [u8] { unsafe { std :: slice :: from_raw_parts (self . mask , (self . capacity + 7) / 8) } } pub fn data_ptr (& mut self) -> * mut u8 { self . data } pub fn mask_ptr (& mut self) -> * mut u8 { self . mask } pub fn epoch (& self) -> Epoch { self . epoch . load () } pub fn set_epoch (& self , epoch : Epoch) { self . epoch . store (epoch) ; } pub fn metadata_entries (& self) -> Vec < (String , Vec < u8 >) > { self . metadata . clone_store () } pub fn set_metadata (& mut self , entries : Vec < (String , Vec < u8 >) >) { self . metadata = Metadata :: from_entries (entries) ; } pub fn metadata_blob (& self) -> Vec < u8 > { let entries = self . metadata . clone_store () ; if entries . is_empty () { return Vec :: new () ; } let mut blob = Vec :: with_capacity (64) ; blob . extend_from_slice (& (entries . len () as u32) . to_le_bytes ()) ; for (key , value) in entries { let key_bytes = key . as_bytes () ; blob . extend_from_slice (& (key_bytes . len () as u32) . to_le_bytes ()) ; blob . extend_from_slice (key_bytes) ; blob . extend_from_slice (& (value . len () as u32) . to_le_bytes ()) ; blob . extend_from_slice (& value) ; } blob } pub fn set_metadata_blob (& mut self , blob : & [u8]) -> Result < () , PageError > { if blob . is_empty () { self . metadata = Metadata :: new () ; return Ok (()) ; } let mut cursor = 0usize ; let entry_count = read_u32 (blob , & mut cursor) ? as usize ; let mut entries = Vec :: with_capacity (entry_count) ; for _ in 0 .. entry_count { let key_len = read_u32 (blob , & mut cursor) ? as usize ; let key_bytes = read_bytes (blob , & mut cursor , key_len) ? ; let value_len = read_u32 (blob , & mut cursor) ? as usize ; let value_bytes = read_bytes (blob , & mut cursor , value_len) ? ; let key = String :: from_utf8 (key_bytes) . map_err (| _ | PageError :: MetadataDecode ("invalid utf-8 key")) ? ; entries . push ((key , value_bytes)) ; } self . metadata = Metadata :: from_entries (entries) ; Ok (()) } pub fn apply_delta (& mut self , delta : & Delta) -> Result < () , PageError > { if let Err (err) = delta_validation :: validate_delta (delta) { return Err (match err { DeltaError :: SizeMismatch { mask_len , payload_len } => PageError :: MaskSizeMismatch { expected : mask_len , found : payload_len , } , DeltaError :: PageIDMismatch { expected , found } => PageError :: PageIDMismatch { expected , found , } , DeltaError :: MaskSizeMismatch { expected , found } => PageError :: MaskSizeMismatch { expected , found , } , }) ; } if delta . page_id != self . id { return Err (PageError :: PageIDMismatch { expected : self . id , found : delta . page_id , }) ; } let mut payload_idx = 0usize ; for i in 0 .. self . capacity { let changed = if i < delta . mask . len () { delta . mask [i] } else { false } ; if changed { if delta . is_sparse { if payload_idx >= delta . payload . len () { return Err (PageError :: MaskSizeMismatch { expected : payload_idx , found : delta . payload . len () , }) ; } unsafe { * self . data . add (i) = delta . payload [payload_idx] ; } payload_idx += 1 ; } else { let payload_i = i . min (delta . payload . len () - 1) ; unsafe { * self . data . add (i) = delta . payload [payload_i] ; } } let mask_byte = unsafe { self . mask . add (i / 8) } ; unsafe { * mask_byte |= 1 << (i % 8) ; } } } self . epoch . store (delta . epoch) ; Ok (()) } } . self_ty` (line 0, priv)
- [Rust | Function] `read_bytes` (line 0, priv)
  - Signature: `fn read_bytes (blob : & [u8] , cursor : & mut usize , len : usize) -> Result < Vec < u8 > , PageError > { if * cursor...`
  - Calls: len, Err, PageError::MetadataDecode, to_vec, Ok
- [Rust | Function] `read_u32` (line 0, priv)
  - Signature: `fn read_u32 (blob : & [u8] , cursor : & mut usize) -> Result < u32 , PageError > { if * cursor + 4 > blob . len () { ...`
  - Calls: len, Err, PageError::MetadataDecode, map_err, try_into, PageError::MetadataDecode, Ok, u32::from_le_bytes

## File: MMSB/src/01_page/replay_validator.rs

- Layer(s): 01_page
- Language coverage: Rust (11)
- Element types: Function (5), Impl (2), Module (1), Struct (3)
- Total elements: 11

### Elements

- [Rust | Struct] `ReplayCheckpoint` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct ReplayCheckpoint { pub id : usize , pub log_offset : u64 , pub snapshot : Vec <...`
- [Rust | Struct] `ReplayReport` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct ReplayReport { pub checkpoint_id : usize , pub divergence : f64 , pub violations : Vec ...`
- [Rust | Struct] `ReplayValidator` (line 0, pub)
  - Signature: `# [derive (Debug)] pub struct ReplayValidator { checkpoints : Vec < ReplayCheckpoint > , threshold : f64 , }`
- [Rust | Function] `checkpoint_validation_detects_divergence` (line 0, priv)
  - Signature: `# [test] fn checkpoint_validation_detects_divergence () { let path = temp_log_path () ; let log = TransactionLog :: n...`
  - Calls: temp_log_path, unwrap, TransactionLog::new, PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, Some, unwrap, allocate_raw, PageID, Some, unwrap, acquire_page, PageID, unwrap, acquire_page, PageID, copy_from_slice, data_mut_slice, set_epoch, Epoch, copy_from_slice, data_mut_slice, ReplayValidator::new, unwrap, record_checkpoint, unwrap, acquire_page, PageID, data_mut_slice, unwrap, validate_allocator, ok, fs::remove_file
- [Rust | Function] `compare_snapshots` (line 0, priv)
  - Signature: `fn compare_snapshots (checkpoint : & ReplayCheckpoint , current : & [PageSnapshotData] ,) -> ReplayReport { let mut b...`
  - Calls: HashMap::new, insert, Vec::new, remove, l2_distance, max, push, keys, push, sqrt
- [Rust | Impl] `impl ReplayReport { pub fn passed (& self , threshold : f64) -> bool { self . divergence <= threshold } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl ReplayValidator { pub fn new (threshold : f64) -> Self { Self { checkpoints : Vec :: new () , threshold , } } pub fn record_checkpoint (& mut self , allocator : & PageAllocator , tlog : & TransactionLog ,) -> std :: io :: Result < usize > { let snapshot = allocator . snapshot_pages () ; let offset = tlog . current_offset () ? ; let id = self . checkpoints . len () ; self . checkpoints . push (ReplayCheckpoint { id , log_offset : offset , snapshot , }) ; Ok (id) } pub fn checkpoint (& self , idx : usize) -> Option < & ReplayCheckpoint > { self . checkpoints . get (idx) } pub fn validate_allocator (& self , checkpoint_id : usize , allocator : & PageAllocator ,) -> Result < ReplayReport , PageError > { let snapshot = allocator . snapshot_pages () ; Ok (self . compare_with_snapshot (checkpoint_id , & snapshot) ?) } pub fn compare_with_snapshot (& self , checkpoint_id : usize , snapshot : & [PageSnapshotData] ,) -> Result < ReplayReport , PageError > { let checkpoint = self . checkpoint (checkpoint_id) . ok_or (PageError :: MetadataDecode ("invalid checkpoint id")) ? ; Ok (compare_snapshots (checkpoint , snapshot)) } pub fn threshold (& self) -> f64 { self . threshold } } . self_ty` (line 0, priv)
- [Rust | Function] `l2_distance` (line 0, priv)
  - Signature: `fn l2_distance (reference : & [u8] , candidate : & [u8]) -> (f64 , u8) { let len = reference . len () . min (candidat...`
  - Calls: min, len, len, max, unsigned_abs, len, len, powi, abs, len, len
- [Rust | Function] `rand_suffix` (line 0, priv)
  - Signature: `fn rand_suffix () -> u64 { use std :: time :: { SystemTime , UNIX_EPOCH } ; SystemTime :: now () . duration_since (UN...`
  - Calls: as_nanos, unwrap, duration_since, SystemTime::now
- [Rust | Function] `temp_log_path` (line 0, priv)
  - Signature: `fn temp_log_path () -> PathBuf { let mut path = std :: env :: temp_dir () ; path . push (format ! ("mmsb_replay_{}.lo...`
  - Calls: std::env::temp_dir, push
- [Rust | Module] `tests` (line 0, priv)

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

