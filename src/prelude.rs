// src/prelude.rs

//! MMSB Core Prelude
//!
//! Layered, agent-readable entrypoint into MMSB.
//!
//! Prefer importing specific layers:
//! ```rust
//! use mmsb_core::prelude::page::*;
//! ```

pub mod types {
    pub use crate::types::*;
}

pub mod page {
    pub use crate::page::{
        load_checkpoint, merge_deltas, validate_delta, write_checkpoint,
        ColumnarDeltaBatch, Delta, DeltaIntegrityChecker,
        DeviceBufferRegistry, HostDeviceSync,
        IntegrityReport, IntegrityViolation, IntegrityViolationKind,
        LockFreeAllocator, LogSummary, Metadata,
        Page, PageAllocator, PageAllocatorConfig,
        PageInfo, PageSnapshotData,
        ReplayCheckpoint, ReplayReport, ReplayValidator,
        TransactionLog, TransactionLogReader,
    };
}

pub mod semiring {
    pub use crate::semiring::{
        Semiring,
        BooleanSemiring,
        TropicalSemiring,
        PurityFailure,
        PurityReport,
        PurityValidator,
        accumulate,
        fold_add,
        fold_mul,
    };
}

pub mod dag {
    pub use crate::dag::{
        Edge, EdgeType,
        has_cycle, topological_sort,
        GraphValidator, GraphValidationReport,
        ShadowPageGraph,
    };
}

pub mod propagation {
    pub use crate::propagation::{
        PropagationCommand,
        PropagationEngine,
        PropagationQueue,
        TickMetrics,
        TickOrchestrator,
        ThroughputEngine,
        ThroughputMetrics,
        passthrough,
    };
}

pub mod adaptive {
    pub use crate::adaptive::{
        AccessPattern,
        LocalityOptimizer,
        MemoryLayout,
        PageCluster,
        PageClusterer,
        PhysAddr,
    };
}

pub mod utility {
    pub use crate::utility::{
        CpuFeatures,
        GCMetrics,
        Invariant,
        InvariantChecker,
        InvariantContext,
        InvariantResult,
        MemoryMonitor,
        MemoryMonitorConfig,
        MemorySnapshot,
        ProvenanceResult,
        ProvenanceTracker,
        Telemetry,
        TelemetrySnapshot,
    };
}

pub mod physical {
    pub use crate::physical::{
        AllocatorStats,
        GPUMemoryPool,
        NCCLContext,
        PoolStats,
    };

    #[cfg(feature = "cuda")]
    pub use crate::physical::{NcclDataType, NcclRedOp};
}
