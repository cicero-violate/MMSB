# Structure Group: src/06_utility

## File: MMSB/src/06_utility/cpu_features.rs

- Layer(s): 06_utility
- Language coverage: Rust (5)
- Element types: Function (3), Impl (1), Struct (1)
- Total elements: 5

### Elements

- [Rust | Struct] `CpuFeatures` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Copy)] pub struct CpuFeatures { pub avx2 : bool , pub avx512f : bool , pub sse42 : bool , ...`
- [Rust | Function] `cpu_has_avx2` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_avx2 () -> bool { CpuFeatures :: get () . avx2 } . sig`
  - Calls: CpuFeatures::get
- [Rust | Function] `cpu_has_avx512` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_avx512 () -> bool { CpuFeatures :: get () . avx512f } . sig`
  - Calls: CpuFeatures::get
- [Rust | Function] `cpu_has_sse42` (line 0, pub)
  - Signature: `# [no_mangle] pub extern "C" fn cpu_has_sse42 () -> bool { CpuFeatures :: get () . sse42 } . sig`
  - Calls: CpuFeatures::get
- [Rust | Impl] `impl CpuFeatures { pub fn detect () -> Self { # [cfg (target_arch = "x86_64")] { Self { avx2 : is_x86_feature_detected ! ("avx2") , avx512f : is_x86_feature_detected ! ("avx512f") , sse42 : is_x86_feature_detected ! ("sse4.2") , bmi2 : is_x86_feature_detected ! ("bmi2") , } } # [cfg (not (target_arch = "x86_64"))] { Self { avx2 : false , avx512f : false , sse42 : false , bmi2 : false , } } } pub fn get () -> & 'static CpuFeatures { CPU_FEATURES . get_or_init (Self :: detect) } } . self_ty` (line 0, priv)

## File: MMSB/src/06_utility/mod.rs

- Layer(s): 06_utility
- Language coverage: Rust (2)
- Element types: Module (2)
- Total elements: 2

### Elements

- [Rust | Module] `cpu_features` (line 0, pub)
- [Rust | Module] `telemetry` (line 0, pub)

## File: MMSB/src/06_utility/telemetry.rs

- Layer(s): 06_utility
- Language coverage: Rust (9)
- Element types: Function (3), Impl (3), Module (1), Struct (2)
- Total elements: 9

### Elements

- [Rust | Impl] `Default for impl Default for Telemetry { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Struct] `Telemetry` (line 0, pub)
  - Signature: `# [doc = " Telemetry counters for system metrics"] # [derive (Debug)] pub struct Telemetry { # [doc = " Total cache m...`
- [Rust | Struct] `TelemetrySnapshot` (line 0, pub)
  - Signature: `# [doc = " Snapshot of telemetry metrics"] # [derive (Debug , Clone , Copy)] pub struct TelemetrySnapshot { pub cache...`
- [Rust | Impl] `impl Telemetry { # [doc = " Create new telemetry tracker"] pub fn new () -> Self { Self { cache_misses : AtomicU64 :: new (0) , cache_hits : AtomicU64 :: new (0) , allocations : AtomicU64 :: new (0) , bytes_allocated : AtomicU64 :: new (0) , propagations : AtomicU64 :: new (0) , propagation_latency_us : AtomicU64 :: new (0) , start_time : Instant :: now () , } } # [doc = " Record cache miss"] pub fn record_cache_miss (& self) { self . cache_misses . fetch_add (1 , Ordering :: Relaxed) ; } # [doc = " Record cache hit"] pub fn record_cache_hit (& self) { self . cache_hits . fetch_add (1 , Ordering :: Relaxed) ; } # [doc = " Record allocation"] pub fn record_allocation (& self , bytes : u64) { self . allocations . fetch_add (1 , Ordering :: Relaxed) ; self . bytes_allocated . fetch_add (bytes , Ordering :: Relaxed) ; } # [doc = " Record propagation"] pub fn record_propagation (& self , latency_us : u64) { self . propagations . fetch_add (1 , Ordering :: Relaxed) ; self . propagation_latency_us . fetch_add (latency_us , Ordering :: Relaxed) ; } # [doc = " Get current snapshot"] pub fn snapshot (& self) -> TelemetrySnapshot { TelemetrySnapshot { cache_misses : self . cache_misses . load (Ordering :: Relaxed) , cache_hits : self . cache_hits . load (Ordering :: Relaxed) , allocations : self . allocations . load (Ordering :: Relaxed) , bytes_allocated : self . bytes_allocated . load (Ordering :: Relaxed) , propagations : self . propagations . load (Ordering :: Relaxed) , propagation_latency_us : self . propagation_latency_us . load (Ordering :: Relaxed) , elapsed_ms : self . start_time . elapsed () . as_millis () as u64 , } } # [doc = " Reset all counters"] pub fn reset (& self) { self . cache_misses . store (0 , Ordering :: Relaxed) ; self . cache_hits . store (0 , Ordering :: Relaxed) ; self . allocations . store (0 , Ordering :: Relaxed) ; self . bytes_allocated . store (0 , Ordering :: Relaxed) ; self . propagations . store (0 , Ordering :: Relaxed) ; self . propagation_latency_us . store (0 , Ordering :: Relaxed) ; } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl TelemetrySnapshot { # [doc = " Compute cache hit rate"] pub fn cache_hit_rate (& self) -> f64 { let total = self . cache_hits + self . cache_misses ; if total == 0 { 0.0 } else { self . cache_hits as f64 / total as f64 } } # [doc = " Compute average propagation latency"] pub fn avg_propagation_latency_us (& self) -> f64 { if self . propagations == 0 { 0.0 } else { self . propagation_latency_us as f64 / self . propagations as f64 } } # [doc = " Compute memory overhead (bytes per allocation)"] pub fn avg_allocation_size (& self) -> f64 { if self . allocations == 0 { 0.0 } else { self . bytes_allocated as f64 / self . allocations as f64 } } } . self_ty` (line 0, priv)
- [Rust | Function] `test_cache_hit_rate` (line 0, priv)
  - Signature: `# [test] fn test_cache_hit_rate () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemetr...`
  - Calls: Telemetry::new, record_cache_hit, record_cache_hit, record_cache_hit, record_cache_miss, snapshot
- [Rust | Function] `test_reset` (line 0, priv)
  - Signature: `# [test] fn test_reset () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemetry . reset...`
  - Calls: Telemetry::new, record_cache_hit, reset, snapshot
- [Rust | Function] `test_telemetry_basic` (line 0, priv)
  - Signature: `# [test] fn test_telemetry_basic () { let telemetry = Telemetry :: new () ; telemetry . record_cache_hit () ; telemet...`
  - Calls: Telemetry::new, record_cache_hit, record_cache_miss, record_allocation, snapshot
- [Rust | Module] `tests` (line 0, priv)

