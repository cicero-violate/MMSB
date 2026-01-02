# Structure Group: src/bin

## File: MMSB/src/bin/phase6_bench.rs

- Layer(s): root
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `build_deltas` (line 0, priv)
  - Signature: `fn build_deltas (count : usize , pages : u64) -> Vec < Delta > { (0 .. count) . map (| idx | { let page_id = PageID (...`
  - Calls: collect, map, PageID, DeltaID, Epoch, Source, into
- [Rust | Function] `build_graph` (line 0, priv)
  - Signature: `fn build_graph (nodes : u64) -> ShadowPageGraph { let graph = ShadowPageGraph :: default () ; for id in 1 .. nodes { ...`
  - Calls: ShadowPageGraph::default, add_edge, PageID, PageID
- [Rust | Function] `main` (line 0, priv)
  - Signature: `fn main () -> Result < () , Box < dyn Error > > { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConf...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, expect, allocate_raw, PageID, Some, build_deltas, unwrap_or, map, std::thread::available_parallelism, get, ThroughputEngine::new, Arc::clone, process_parallel, clone, Arc::new, build_graph, Arc::new, MemoryMonitor::with_config, Arc::clone, MemoryMonitorConfig::default, ThroughputEngine::new, Arc::clone, TickOrchestrator::new, Arc::clone, execute_tick, write_report, Ok
- [Rust | Function] `write_report` (line 0, priv)
  - Signature: `fn write_report (throughput : & mmsb_core :: propagation :: ThroughputMetrics , tick : & mmsb_core :: propagation :: ...`
  - Calls: as_secs_f64, duration_since, SystemTime::now, File::create, Ok

