# Structure Group: src/06_utility

## File: MMSB/src/06_utility/CostAggregation.jl

- Layer(s): 06_utility
- Language coverage: Julia (4)
- Element types: Function (2), Module (1), Struct (1)
- Total elements: 4

### Elements

- [Julia | Module] `CostAggregation` (line 7, pub)
- [Julia | Struct] `WeightedCost` (line 16, pub)
  - Signature: `struct WeightedCost`
- [Julia | Function] `aggregate_costs` (line 27, pub)
  - Signature: `aggregate_costs(costs::Vector{WeightedCost})`
  - Calls: isempty, sum
- [Julia | Function] `normalize_costs` (line 37, pub)
  - Signature: `normalize_costs(costs::Vector{WeightedCost})`
  - Calls: WeightedCost, else, haskey, isempty, maximum, minimum, push!

## File: MMSB/src/06_utility/ErrorRecovery.jl

- Layer(s): 06_utility
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `ErrorRecovery` (line 1, pub)
- [Julia | Struct] `RetryPolicy` (line 5, pub)
  - Signature: `struct RetryPolicy`
- [Julia | Function] `exponential_backoff` (line 12, pub)
  - Signature: `exponential_backoff(a,p)`
  - Calls: min, round
- [Julia | Function] `is_retryable_error` (line 13, pub)
  - Signature: `is_retryable_error(e)`
  - Calls: ccall
- [Julia | Function] `is_fatal_error` (line 14, pub)
  - Signature: `is_fatal_error(e)`
  - Calls: ccall
- [Julia | Function] `retry_with_backoff` (line 15, pub)
  - Signature: `retry_with_backoff(f,p`
  - Calls: RetryPolicy

## File: MMSB/src/06_utility/MemoryPressure.jl

- Layer(s): 06_utility
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `MemoryPressure` (line 1, pub)
- [Julia | Struct] `LRUTracker` (line 5, pub)
  - Signature: `mutable struct LRUTracker`
- [Julia | Function] `LRUTracker` (line 9, pub)
  - Signature: `LRUTracker()`
  - Calls: LRUTracker
- [Julia | Function] `record_access` (line 11, pub)
  - Signature: `record_access(s,p)`
  - Calls: LRUTracker, get!
- [Julia | Function] `evict_lru_pages` (line 13, pub)
  - Signature: `evict_lru_pages(s,n)`
  - Calls: LRUTracker, collect, delete!, get!, length, min, push!, sort

## File: MMSB/src/06_utility/Monitoring.jl

- Layer(s): 06_utility
- Language coverage: Julia (9)
- Element types: Function (7), Module (1), Struct (1)
- Total elements: 9

### Elements

- [Julia | Module] `Monitoring` (line 8, pub)
- [Julia | Struct] `MMSBStats` (line 17, pub)
  - Signature: `mutable struct MMSBStats`
- [Julia | Function] `track_delta_latency!` (line 40, pub)
  - Signature: `track_delta_latency!(state::MMSBState, duration_ns::UInt64)`
  - Calls: UInt64, get
- [Julia | Function] `track_propagation_latency!` (line 45, pub)
  - Signature: `track_propagation_latency!(state::MMSBState, duration_ns::UInt64)`
  - Calls: UInt64, get
- [Julia | Function] `compute_graph_depth` (line 50, pub)
  - Signature: `compute_graph_depth(graph)`
  - Calls: _dfs_depth, max
- [Julia | Function] `_dfs_depth` (line 59, pub)
  - Signature: `_dfs_depth(graph, node::PageID, depth::Int, visited::Set{PageID})`
  - Calls: _dfs_depth, delete!, get, isempty, maximum, push!
- [Julia | Function] `get_stats` (line 74, pub)
  - Signature: `get_stats(state::MMSBState)::MMSBStats`
  - Calls: FFIWrapper.rust_tlog_summary, Int64, MMSBStats, UInt64, compute_graph_depth, get, length, lock, sum, values
- [Julia | Function] `print_stats` (line 108, pub)
  - Signature: `print_stats(state::MMSBState)`
  - Calls: get_stats, println, round
- [Julia | Function] `reset_stats!` (line 119, pub)
  - Signature: `reset_stats!(state::MMSBState)`
  - Calls: UInt64, haskey

## File: MMSB/src/06_utility/cost_functions.jl

- Layer(s): 06_utility
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `CostFunctions` (line 9, pub)
- [Julia | Struct] `CostComponents` (line 18, pub)
  - Signature: `struct CostComponents`
- [Julia | Function] `compute_cache_cost` (line 30, pub)
  - Signature: `compute_cache_cost(cache_misses::Int, cache_hits::Int)`
- [Julia | Function] `compute_memory_cost` (line 45, pub)
  - Signature: `compute_memory_cost(bytes_allocated::Int, num_allocations::Int)`
- [Julia | Function] `compute_latency_cost` (line 59, pub)
  - Signature: `compute_latency_cost(total_latency_us::Int, num_ops::Int)`
- [Julia | Function] `from_telemetry` (line 73, pub)
  - Signature: `from_telemetry(snapshot)`
  - Calls: CostComponents, Float64, Int, compute_cache_cost, compute_latency_cost, compute_memory_cost

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

## File: MMSB/src/06_utility/entropy_measure.jl

- Layer(s): 06_utility
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `EntropyMeasure` (line 9, pub)
- [Julia | Struct] `PageDistribution` (line 18, pub)
  - Signature: `struct PageDistribution`
- [Julia | Function] `PageDistribution` (line 23, pub)
  - Signature: `PageDistribution(counts::Dict{UInt64, Int})`
  - Calls: PageDistribution, sum, values
- [Julia | Function] `compute_entropy` (line 32, pub)
  - Signature: `compute_entropy(dist::PageDistribution)`
  - Calls: log2, values
- [Julia | Function] `state_entropy` (line 51, pub)
  - Signature: `state_entropy(access_pattern::Dict{Tuple{UInt64, UInt64}, Int})`
  - Calls: isempty, log2, sum, values
- [Julia | Function] `entropy_reduction` (line 71, pub)
  - Signature: `entropy_reduction(old_entropy::Float64, new_entropy::Float64)`
  - Calls: return

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

## File: MMSB/src/06_utility/utility_engine.jl

- Layer(s): 06_utility
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `UtilityEngine` (line 9, pub)
- [Julia | Struct] `UtilityState` (line 20, pub)
  - Signature: `mutable struct UtilityState`
- [Julia | Function] `UtilityState` (line 27, pub)
  - Signature: `UtilityState(max_history::Int`
- [Julia | Function] `compute_utility` (line 44, pub)
  - Signature: `compute_utility(costs::CostComponents, weights::Dict{Symbol, Float64})`
- [Julia | Function] `update_utility!` (line 58, pub)
  - Signature: `update_utility!(state::UtilityState, costs::CostComponents)`
  - Calls: compute_utility, length, popfirst!, push!
- [Julia | Function] `utility_trend` (line 75, pub)
  - Signature: `utility_trend(state::UtilityState)`
  - Calls: abs, length, mean

