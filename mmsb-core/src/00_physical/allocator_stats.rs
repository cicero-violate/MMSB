use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct AllocatorStats {
    allocations: AtomicU64,
    frees: AtomicU64,
}

impl AllocatorStats {
    pub fn record_alloc(&self) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_free(&self) {
        self.frees.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.allocations.load(Ordering::Relaxed),
            self.frees.load(Ordering::Relaxed),
        )
    }
}
