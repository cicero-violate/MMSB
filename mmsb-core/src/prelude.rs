// src/prelude.rs

//! MMSB Core Prelude
//!
//! Layered, agent-readable entrypoint into MMSB.
//!
//! Prefer importing specific layers:
//! ```rust
//! use mmsb_core::prelude::page::*;
//! ```

// Rule: Root-level re-exports must be STABLE NOUNS ONLY

// Re-export commonly used types at root level for convenience
pub use crate::types::{
    DeltaID, Epoch, PageID, PageLocation, Source, MemoryPressureHandler,
};
pub use crate::page::{
    load_checkpoint, merge_deltas, validate_delta, write_checkpoint,
    Delta, DeltaIntegrityChecker, DeviceBufferRegistry,
    LockFreeAllocator, Page, PageAllocator, PageAllocatorConfig,
    ReplayValidator, TransactionLog, TransactionLogReader,
};
pub use crate::dag::{
    EdgeType, GraphValidator, ShadowPageGraph,
    DependencyGraph, StructuralOp, DagCommitError, commit_structural_delta,
};
pub use crate::propagation::{
    passthrough, PropagationCommand, PropagationEngine, PropagationQueue,
    ThroughputEngine, ThroughputMetrics, TickOrchestrator, TickMetrics,
};
pub use crate::utility::{
    CpuFeatures,
    InvariantChecker, InvariantContext,
    MemoryMonitor, MemoryMonitorConfig,
    ProvenanceTracker,
};

pub use crate::proof::{
    MmsbAdmissionProof, MmsbStructuralAdmissionProof,
    MmsbExecutionProof, ADMISSION_PROOF_VERSION, EXECUTION_PROOF_VERSION, STRUCTURAL_PROOF_VERSION,
};

pub use crate::materialization::{materialize_page_state, MaterializedPageView};

pub use crate::semiring::{BooleanSemiring, PurityValidator, TropicalSemiring, fold_add};

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
        apply_log, compact, generate_mask, read_log, summary,
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
        DependencyGraph, StructuralOp, DagCommitError, commit_structural_delta,
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

pub mod materialization {
    pub use crate::materialization::{materialize_page_state, MaterializedPageView};
}

pub mod proof {
    pub use crate::proof::*;
}

pub mod utility {
    pub use crate::utility::*;
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
