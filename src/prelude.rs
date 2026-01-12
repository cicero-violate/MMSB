// src/prelude.rs
//! MMSB Core Prelude
//!
//! The prelude module re-exports the primary types, traits, and functions
//! that users need when working with mmsb-core.
//!
//! ```rust
//! use mmsb_core::prelude::*;
//! ```

// Core types
pub use crate::types::{
    DeltaError, DeltaID, Epoch, EpochCell, PageError, PageID, PageLocation, Source,
    MemoryPressureHandler,
};

// Page management
pub use crate::page::{
    load_checkpoint, merge_deltas, validate_delta, write_checkpoint, ColumnarDeltaBatch, Delta,
    DeltaIntegrityChecker, HostDeviceSync, IntegrityReport, IntegrityViolation,
    IntegrityViolationKind, LogSummary, Metadata, Page, PageAllocator, PageAllocatorConfig,
    PageInfo, PageSnapshotData, ReplayCheckpoint, ReplayReport, ReplayValidator, TransactionLog,
    TransactionLogReader,
};

// Semiring abstractions
pub use crate::semiring::{
    accumulate, fold_add, fold_mul, BooleanSemiring, PurityFailure, PurityReport,
    PurityValidator, Semiring, TropicalSemiring,
};

// Dependency graph
pub use crate::dag::{
    has_cycle, topological_sort, Edge, EdgeType, GraphValidationReport, GraphValidator,
    ShadowPageGraph,
};

// Propagation engine
pub use crate::propagation::{
    passthrough, PropagationCommand, PropagationEngine, PropagationQueue, TickMetrics,
    TickOrchestrator, ThroughputEngine, ThroughputMetrics,
};

// Adaptive memory
pub use crate::adaptive::{
    AccessPattern, LocalityOptimizer, MemoryLayout, PageCluster, PageClusterer, PhysAddr,
};

// Utility and monitoring
pub use crate::utility::{
    CpuFeatures, GCMetrics, Invariant, InvariantChecker, InvariantContext, InvariantResult,
    MemoryMonitor, MemoryMonitorConfig, MemorySnapshot, ProvenanceResult, ProvenanceTracker,
    Telemetry, TelemetrySnapshot,
};

// Physical layer (GPU/CUDA support)
pub use crate::physical::{AllocatorStats, GPUMemoryPool, NCCLContext, PoolStats};

#[cfg(feature = "cuda")]
pub use crate::physical::{NcclDataType, NcclRedOp};
