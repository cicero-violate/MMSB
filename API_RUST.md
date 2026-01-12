## **MMSB Rust Library API Reference**

### **Cargo.toml Setup**

```toml
[dependencies]
mmsb-core = { path = "/path/to/MMSB" }
# or when published:
# mmsb-core = "0.1.0"

# Optional: enable CUDA support
# mmsb-core = { version = "0.1.0", features = ["cuda"] }
```

### **Module Structure**

```rust
use mmsb_core::*;  // Import all public APIs

// Or import specific modules:
use mmsb_core::page::*;
use mmsb_core::types::*;
use mmsb_core::semiring::*;
use mmsb_core::dag::*;
use mmsb_core::propagation::*;
use mmsb_core::adaptive::*;
use mmsb_core::utility::*;
use mmsb_core::physical::*;
```

---

### **Core Types Module (`types`)**

#### **PageID and DeltaID**
```rust
let page_id = PageID(42);
let delta_id = DeltaID(100);
```

#### **Epoch (Logical Timestamp)**
```rust
let epoch = Epoch(1);
let epoch_cell = EpochCell::new(0);
epoch_cell.increment();
let current = epoch_cell.get();
```

#### **PageLocation**
```rust
// Where the page is stored
enum PageLocation {
    Cpu,      // Host memory
    Gpu,      // Device memory
    Unified,  // CUDA unified memory
}

let loc = PageLocation::Cpu;
```

#### **Source (Delta Origin)**
```rust
// Tracks where delta originated
enum Source {
    Disk,
    Network,
    Compute,
    User,
    Replay,
    Merge,
}
```

---

### **Page Management Module (`page`)**

#### **PageAllocator - Core Memory Manager**

```rust
use mmsb_core::page::{PageAllocator, PageAllocatorConfig, PageLocation};

// Create allocator
let config = PageAllocatorConfig {
    default_location: PageLocation::Cpu,
};
let allocator = PageAllocator::new(config);

// Allocate a page
let page_id = PageID(1);
let size = 4096;
let page = allocator
    .allocate(page_id, size, Some(PageLocation::Unified))
    .expect("Failed to allocate page");

// Get existing page
let page_ref = allocator.get_page(page_id).expect("Page not found");

// Get page count
let count = allocator.page_count();

// List all pages
let pages: Vec<PageInfo> = allocator.list_pages();

// Release a page
allocator.release(page_id);

// Clear all pages
allocator.clear();
```

#### **Page - Memory Buffer**

```rust
use mmsb_core::page::{Page, Metadata};

// Create page directly (usually via allocator)
let page = Page::new(PageID(1), 4096, PageLocation::Cpu)?;

// Read data
let data_slice: &[u8] = page.data_slice();
let mask_slice: &[u8] = page.mask_slice();

// Write data (mutable)
let mut page = page;
let data_mut: &mut [u8] = page.data_mut_slice();
data_mut[0] = 42;

// Get page properties
let size = page.size();
let location = page.location();
let page_id = page.id();

// Epoch management
let current_epoch = page.epoch();
page.advance_epoch();

// Metadata (key-value store)
let metadata = Metadata::new();
metadata.insert("key", vec![1, 2, 3]);
let entries = metadata.clone_store();
```

#### **Delta - State Change**

```rust
use mmsb_core::page::{Delta, DeltaID, Source};

// Create dense delta (all bytes specified)
let mask = vec![true, true, false, false];  // First 2 bytes change
let payload = vec![10, 20, 0, 0];
let delta = Delta::new_dense(
    DeltaID(1),
    PageID(42),
    Epoch(1),
    payload,
    mask,
    Source::User,
)?;

// Create sparse delta (only changed bytes in payload)
let mask = vec![true, false, true, false];  // Bytes 0 and 2 change
let payload = vec![10, 20];  // Only 2 values for changed bytes
let delta = Delta::new_sparse(
    DeltaID(2),
    PageID(42),
    Epoch(2),
    mask,
    payload,
    Source::Compute,
)?;

// Merge deltas
let merged = delta1.merge(&delta2)?;

// Apply delta to page
page.apply_delta(&delta)?;

// Convert to dense representation
let dense_payload = delta.to_dense();

// Access delta properties
let page_id = delta.page_id;
let epoch = delta.epoch;
let is_sparse = delta.is_sparse;
let timestamp = delta.timestamp;
```

#### **TransactionLog - Persistent State**

```rust
use mmsb_core::page::{TransactionLog, TransactionLogReader};

// Create/open transaction log
let tlog = TransactionLog::new("state.tlog")?;

// Append delta
tlog.append(delta)?;

// Get log length
let count = tlog.len();

// Drain all entries (consumes)
let deltas: Vec<Delta> = tlog.drain();

// Get log summary
let summary = mmsb_core::page::summary("state.tlog")?;
println!("Total deltas: {}", summary.total_deltas);
println!("Total bytes: {}", summary.total_bytes);
println!("Last epoch: {}", summary.last_epoch);

// Read log sequentially
let mut reader = TransactionLogReader::open("state.tlog")?;
while let Some(delta) = reader.next()? {
    println!("Read delta: {:?}", delta);
}
```

#### **Checkpoint - Snapshot State**

```rust
use mmsb_core::page::{write_checkpoint, load_checkpoint};

// Write checkpoint
write_checkpoint(&allocator, &tlog, "snapshot.chk")?;

// Load checkpoint
let allocator = PageAllocator::new(config);
let tlog = TransactionLog::new("state.tlog")?;
load_checkpoint(&allocator, &tlog, "snapshot.chk")?;

// After load, allocator contains all pages at checkpoint epoch
let page = allocator.get_page(PageID(42))?;
```

---

### **Semiring Module (`semiring`)**

#### **Semiring Trait**

```rust
use mmsb_core::semiring::Semiring;

pub trait Semiring: Send + Sync {
    type Element: Clone + PartialEq + Send + Sync;
    
    fn zero(&self) -> Self::Element;     // Additive identity
    fn one(&self) -> Self::Element;      // Multiplicative identity
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}
```

#### **Built-in Semirings**

```rust
use mmsb_core::semiring::{TropicalSemiring, BooleanSemiring};

// Tropical semiring (min, +, ∞, 0)
let tropical = TropicalSemiring;
let min_val = tropical.add(&5.0, &3.0);  // Returns 3.0
let sum_val = tropical.mul(&5.0, &3.0);  // Returns 8.0

// Boolean semiring (OR, AND, false, true)
let boolean = BooleanSemiring;
let or_val = boolean.add(&true, &false);   // Returns true
let and_val = boolean.mul(&true, &false);  // Returns false
```

#### **Semiring Operations**

```rust
use mmsb_core::semiring::{accumulate, fold_add, fold_mul};

// Accumulate values using semiring
let values = vec![1.0, 2.0, 3.0];
let result = accumulate(&tropical, &values);

// Fold with addition
let sum = fold_add(&tropical, &values);

// Fold with multiplication
let product = fold_mul(&tropical, &values);
```

---

### **DAG Module (`dag`)**

#### **ShadowPageGraph - Dependency Tracking**

```rust
use mmsb_core::dag::{ShadowPageGraph, EdgeType, Edge};

// Create graph
let graph = ShadowPageGraph::default();

// Add dependency edges
graph.add_edge(PageID(1), PageID(2), EdgeType::Data);
graph.add_edge(PageID(1), PageID(3), EdgeType::Control);

// Remove edge
graph.remove_edge(PageID(1), PageID(2));

// Get all descendants of a page
let descendants = graph.descendants(PageID(1));

// Check for cycles
use mmsb_core::dag::has_cycle;
if has_cycle(&graph) {
    panic!("Cycle detected in dependency graph!");
}

// Topological sort
use mmsb_core::dag::topological_sort;
let order = topological_sort(&graph)?;
```

#### **EdgeType**

```rust
enum EdgeType {
    Data,        // Data dependency
    Control,     // Control flow dependency
    Metadata,    // Metadata dependency
}
```

#### **Graph Validation**

```rust
use mmsb_core::dag::GraphValidator;

let validator = GraphValidator::new(&graph);
let report = validator.validate();

if !report.is_valid {
    for error in report.errors {
        eprintln!("Graph error: {:?}", error);
    }
}
```

---

### **Propagation Module (`propagation`)**

#### **PropagationEngine - Event System**

```rust
use mmsb_core::propagation::{PropagationEngine, PropagationCommand};
use std::sync::Arc;

// Create engine
let engine = PropagationEngine::default();

// Register callback for page updates
let page_id = PageID(42);
engine.register_callback(page_id, Arc::new(|page, dependencies| {
    println!("Page {} updated!", page.id().0);
    println!("Depends on {} pages", dependencies.len());
    
    // Perform propagation logic here
    // e.g., recompute derived values using semiring operations
}));

// Enqueue propagation command
let command = PropagationCommand {
    page_id,
    page: Arc::new(page),
    dependencies: vec![Arc::new(dep_page1), Arc::new(dep_page2)],
};
engine.enqueue(command);

// Process all pending propagations
engine.drain();
```

#### **TickOrchestrator - Epoch Coordination**

```rust
use mmsb_core::propagation::TickOrchestrator;

let orchestrator = TickOrchestrator::new();

// Execute one tick (epoch)
orchestrator.tick();

// Get metrics
let metrics = orchestrator.metrics();
println!("Ticks executed: {}", metrics.total_ticks);
println!("Pages propagated: {}", metrics.pages_propagated);
```

#### **ThroughputEngine - Performance Monitoring**

```rust
use mmsb_core::propagation::ThroughputEngine;

let engine = ThroughputEngine::new();

// Record operation
engine.record_operation(/* ... */);

// Get metrics
let metrics = engine.metrics();
println!("Ops/sec: {}", metrics.operations_per_second);
println!("Avg latency: {}μs", metrics.average_latency_us);
```

---

### **Adaptive Module (`adaptive`)**

#### **PageClusterer - Locality Optimization**

```rust
use mmsb_core::adaptive::{PageClusterer, PageCluster};

let clusterer = PageClusterer::new();

// Add pages to cluster based on access patterns
clusterer.add_page(PageID(1));
clusterer.add_page(PageID(2));

// Get clusters
let clusters: Vec<PageCluster> = clusterer.get_clusters();

for cluster in clusters {
    println!("Cluster: {:?}", cluster.page_ids);
}
```

#### **LocalityOptimizer**

```rust
use mmsb_core::adaptive::LocalityOptimizer;

let optimizer = LocalityOptimizer::new();

// Optimize memory layout based on access patterns
optimizer.optimize(&allocator);
```

---

### **Utility Module (`utility`)**

#### **Telemetry - Monitoring**

```rust
use mmsb_core::utility::Telemetry;

let telemetry = Telemetry::new();

// Record metrics
telemetry.record_delta_apply(/* duration */);
telemetry.record_propagation(/* duration */);

// Get snapshot
let snapshot = telemetry.snapshot();
println!("Total delta applies: {}", snapshot.delta_applies);
println!("Avg propagation time: {}μs", snapshot.avg_propagation_us);
```

#### **InvariantChecker - Correctness Validation**

```rust
use mmsb_core::utility::{InvariantChecker, InvariantContext};

let checker = InvariantChecker::new();

// Check invariants
let context = InvariantContext {
    allocator: &allocator,
    graph: &graph,
    current_epoch: Epoch(5),
};

let results = checker.check_all(&context);

for result in results {
    if !result.passed {
        eprintln!("Invariant violation: {}", result.message);
    }
}
```

#### **MemoryMonitor - Memory Tracking**

```rust
use mmsb_core::utility::{MemoryMonitor, MemoryMonitorConfig};

let config = MemoryMonitorConfig::default();
let monitor = MemoryMonitor::new(config);

// Take snapshot
let snapshot = monitor.snapshot();
println!("Used memory: {} MB", snapshot.used_bytes / 1_000_000);
println!("Peak memory: {} MB", snapshot.peak_bytes / 1_000_000);
```

---

### **Physical Module (`physical`)**

#### **GPUMemoryPool - Device Memory Management**

```rust
use mmsb_core::physical::GPUMemoryPool;

let pool = GPUMemoryPool::new()?;

// Allocate GPU memory
let ptr = pool.allocate(1024 * 1024)?;  // 1 MB

// Get pool statistics
let stats = pool.stats();
println!("Total allocated: {} bytes", stats.total_allocated);
println!("Peak usage: {} bytes", stats.peak_usage);

// Free memory
pool.free(ptr)?;
```

#### **NCCLContext - Multi-GPU Communication**

```rust
use mmsb_core::physical::{NCCLContext, NcclDataType, NcclRedOp};

let context = NCCLContext::new(/* gpu_ids */)?;

// Perform all-reduce
context.all_reduce(
    send_buffer,
    recv_buffer,
    count,
    NcclDataType::Float32,
    NcclRedOp::Sum,
)?;
```

---

### **Complete Usage Example**

```rust
use mmsb_core::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup allocator
    let config = page::PageAllocatorConfig {
        default_location: types::PageLocation::Cpu,
    };
    let allocator = page::PageAllocator::new(config);
    
    // 2. Create transaction log
    let tlog = page::TransactionLog::new("state.tlog")?;
    
    // 3. Allocate pages
    let page1 = allocator.allocate(types::PageID(1), 4096, None)?;
    let page2 = allocator.allocate(types::PageID(2), 4096, None)?;
    
    // 4. Create dependency graph
    let graph = dag::ShadowPageGraph::default();
    graph.add_edge(types::PageID(1), types::PageID(2), dag::EdgeType::Data);
    
    // 5. Setup propagation engine
    let engine = propagation::PropagationEngine::default();
    
    // Register callback for page 2
    engine.register_callback(types::PageID(2), std::sync::Arc::new(|page, deps| {
        println!("Page 2 updated! Recalculating...");
        // Propagation logic here
    }));
    
    // 6. Create and apply delta
    let mask = vec![true; 4096];
    let payload = vec![42u8; 4096];
    let delta = page::Delta::new_dense(
        types::DeltaID(1),
        types::PageID(1),
        types::Epoch(1),
        payload,
        mask,
        types::Source::User,
    )?;
    
    // Log delta
    tlog.append(delta.clone())?;
    
    // Apply to page
    let mut page1_mut = allocator.get_page_mut(types::PageID(1))?;
    page1_mut.apply_delta(&delta)?;
    
    // 7. Trigger propagation
    let command = propagation::PropagationCommand {
        page_id: types::PageID(2),
        page: std::sync::Arc::new(page1_mut.clone()),
        dependencies: vec![],
    };
    engine.enqueue(command);
    engine.drain();
    
    // 8. Checkpoint
    page::write_checkpoint(&allocator, &tlog, "snapshot.chk")?;
    
    println!("State successfully saved!");
    
    Ok(())
}
```

---

### **Error Handling**

All major operations return `Result` types:

```rust
// Page operations
pub enum PageError {
    InvalidSize(usize),
    AlreadyExists(PageID),
    NotFound(PageID),
    AllocationFailed(String),
    // ...
}

// Delta operations
pub enum DeltaError {
    SizeMismatch { mask_len: usize, payload_len: usize },
    PageIDMismatch { expected: PageID, found: PageID },
    MaskSizeMismatch { expected: usize, found: usize },
    // ...
}
```

This is the complete public Rust API for using MMSB as a library!
