# CFG Group: src/bin

## Function: `build_deltas`

- File: MMSB/src/bin/phase6_bench.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    build_deltas_0["ENTRY"]
    build_deltas_1["(0 .. count) . map (| idx | { let page_id = PageID (((idx as u64) % pages) + ..."]
    build_deltas_2["EXIT"]
    build_deltas_0 --> build_deltas_1
    build_deltas_1 --> build_deltas_2
```

## Function: `build_graph`

- File: MMSB/src/bin/phase6_bench.rs
- Branches: 0
- Loops: 1
- Nodes: 7
- Edges: 7

```mermaid
flowchart TD
    build_graph_0["ENTRY"]
    build_graph_1["let graph = ShadowPageGraph :: default ()"]
    build_graph_2["for id in 1 .. nodes"]
    build_graph_3["graph . add_edge (PageID (id) , PageID (id + 1) , EdgeType :: Data)"]
    build_graph_4["after for"]
    build_graph_5["graph"]
    build_graph_6["EXIT"]
    build_graph_0 --> build_graph_1
    build_graph_1 --> build_graph_2
    build_graph_2 --> build_graph_3
    build_graph_3 --> build_graph_2
    build_graph_2 --> build_graph_4
    build_graph_4 --> build_graph_5
    build_graph_5 --> build_graph_6
```

## Function: `main`

- File: MMSB/src/bin/phase6_bench.rs
- Branches: 0
- Loops: 1
- Nodes: 19
- Edges: 19

```mermaid
flowchart TD
    main_0["ENTRY"]
    main_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    main_2["let page_count : u64 = 64"]
    main_3["for id in 1 ..= page_count"]
    main_4["allocator . allocate_raw (PageID (id) , 4096 , Some (PageLocation :: Cpu)) . ..."]
    main_5["after for"]
    main_6["let deltas = build_deltas (20_000 , page_count as u64)"]
    main_7["let worker_count = std :: thread :: available_parallelism () . map (| n | n . get ()) . unwrap_o..."]
    main_8["let throughput_engine = ThroughputEngine :: new (Arc :: clone (& allocator) , worker_count , 1024)"]
    main_9["let throughput_metrics = throughput_engine . process_parallel (deltas . clone ()) ?"]
    main_10["let graph = Arc :: new (build_graph (page_count))"]
    main_11["let memory_monitor : Arc < dyn MemoryPressureHandler > = Arc :: new (MemoryMonitor :: with_config (Arc :: clone (& allocator) , Memory..."]
    main_12["let tick_throughput = ThroughputEngine :: new (Arc :: clone (& allocator) , worker_count , 1024)"]
    main_13["let orchestrator = TickOrchestrator :: new (tick_throughput , Arc :: clone (& graph) , memory_mo..."]
    main_14["let tick_metrics = orchestrator . execute_tick (deltas) ?"]
    main_15["write_report (& throughput_metrics , & tick_metrics , worker_count) ?"]
    main_16["macro println"]
    main_17["Ok (())"]
    main_18["EXIT"]
    main_0 --> main_1
    main_1 --> main_2
    main_2 --> main_3
    main_3 --> main_4
    main_4 --> main_3
    main_3 --> main_5
    main_5 --> main_6
    main_6 --> main_7
    main_7 --> main_8
    main_8 --> main_9
    main_9 --> main_10
    main_10 --> main_11
    main_11 --> main_12
    main_12 --> main_13
    main_13 --> main_14
    main_14 --> main_15
    main_15 --> main_16
    main_16 --> main_17
    main_17 --> main_18
```

## Function: `write_report`

- File: MMSB/src/bin/phase6_bench.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    write_report_0["ENTRY"]
    write_report_1["let timestamp = SystemTime :: now () . duration_since (UNIX_EPOCH) ? . as_secs_f64 ()"]
    write_report_2["let mut file = File :: create ('benchmark/results/phase6.json') ?"]
    write_report_3["writeln ! (file , r#'{{ 'timestamp': '{timestamp}', 'throughput': {{ 'process..."]
    write_report_4["Ok (())"]
    write_report_5["EXIT"]
    write_report_0 --> write_report_1
    write_report_1 --> write_report_2
    write_report_2 --> write_report_3
    write_report_3 --> write_report_4
    write_report_4 --> write_report_5
```

