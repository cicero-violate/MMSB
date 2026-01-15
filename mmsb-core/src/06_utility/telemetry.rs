//! Telemetry collection for utility computation
//!
//! Tracks system metrics used to compute cost functions and utility scores.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Telemetry counters for system metrics
#[derive(Debug)]
pub struct Telemetry {
    /// Total cache misses
    pub cache_misses: AtomicU64,
    /// Total cache hits
    pub cache_hits: AtomicU64,
    /// Total memory allocations
    pub allocations: AtomicU64,
    /// Total bytes allocated
    pub bytes_allocated: AtomicU64,
    /// Total propagation operations
    pub propagations: AtomicU64,
    /// Total propagation latency (microseconds)
    pub propagation_latency_us: AtomicU64,
    /// Start timestamp
    start_time: Instant,
}

/// Snapshot of telemetry metrics
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySnapshot {
    pub cache_misses: u64,
    pub cache_hits: u64,
    pub allocations: u64,
    pub bytes_allocated: u64,
    pub propagations: u64,
    pub propagation_latency_us: u64,
    pub elapsed_ms: u64,
}

impl Telemetry {
    /// Create new telemetry tracker
    pub fn new() -> Self {
        Self {
            cache_misses: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
            bytes_allocated: AtomicU64::new(0),
            propagations: AtomicU64::new(0),
            propagation_latency_us: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Record cache miss
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record cache hit
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record allocation
    pub fn record_allocation(&self, bytes: u64) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.bytes_allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record propagation
    pub fn record_propagation(&self, latency_us: u64) {
        self.propagations.fetch_add(1, Ordering::Relaxed);
        self.propagation_latency_us.fetch_add(latency_us, Ordering::Relaxed);
    }

    /// Get current snapshot
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            allocations: self.allocations.load(Ordering::Relaxed),
            bytes_allocated: self.bytes_allocated.load(Ordering::Relaxed),
            propagations: self.propagations.load(Ordering::Relaxed),
            propagation_latency_us: self.propagation_latency_us.load(Ordering::Relaxed),
            elapsed_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.cache_misses.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
        self.bytes_allocated.store(0, Ordering::Relaxed);
        self.propagations.store(0, Ordering::Relaxed);
        self.propagation_latency_us.store(0, Ordering::Relaxed);
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySnapshot {
    /// Compute cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Compute average propagation latency
    pub fn avg_propagation_latency_us(&self) -> f64 {
        if self.propagations == 0 {
            0.0
        } else {
            self.propagation_latency_us as f64 / self.propagations as f64
        }
    }

    /// Compute memory overhead (bytes per allocation)
    pub fn avg_allocation_size(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.bytes_allocated as f64 / self.allocations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_basic() {
        let telemetry = Telemetry::new();
        telemetry.record_cache_hit();
        telemetry.record_cache_miss();
        telemetry.record_allocation(4096);

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.allocations, 1);
        assert_eq!(snapshot.bytes_allocated, 4096);
    }

    #[test]
    fn test_cache_hit_rate() {
        let telemetry = Telemetry::new();
        telemetry.record_cache_hit();
        telemetry.record_cache_hit();
        telemetry.record_cache_hit();
        telemetry.record_cache_miss();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cache_hit_rate(), 0.75);
    }

    #[test]
    fn test_reset() {
        let telemetry = Telemetry::new();
        telemetry.record_cache_hit();
        telemetry.reset();

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.cache_hits, 0);
    }
}
