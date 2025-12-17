use crate::page::PageAllocator;
use crate::physical::AllocatorStats;
use crate::types::PageID;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub total_pages: usize,
    pub total_bytes: usize,
    pub avg_bytes_per_page: usize,
    pub cold_pages: Vec<PageID>,
    pub collected_at: Instant,
}

#[derive(Debug, Clone, Copy)]
pub struct GCMetrics {
    pub reclaimed_pages: usize,
    pub reclaimed_bytes: usize,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct MemoryMonitorConfig {
    pub gc_threshold_bytes: usize,
    pub cold_page_age_limit: u64,
    pub incremental_batch_pages: usize,
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            gc_threshold_bytes: 1 * 1024 * 1024 * 1024, // 1GB
            cold_page_age_limit: 3,
            incremental_batch_pages: 32,
        }
    }
}

struct PageAging {
    last_epoch: u32,
    age: u64,
}

pub struct MemoryMonitor {
    allocator: Arc<PageAllocator>,
    stats: Arc<AllocatorStats>,
    config: MemoryMonitorConfig,
    aging: Mutex<HashMap<PageID, PageAging>>,
}

impl MemoryMonitor {
    pub fn new(allocator: Arc<PageAllocator>) -> Self {
        Self::with_config(allocator, MemoryMonitorConfig::default())
    }

    pub fn with_config(
        allocator: Arc<PageAllocator>,
        config: MemoryMonitorConfig,
    ) -> Self {
        let stats = allocator.stats();
        Self {
            allocator,
            stats,
            config,
            aging: Mutex::new(HashMap::new()),
        }
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        let infos = self.allocator.page_infos();
        self.build_snapshot(&infos)
    }

    pub fn trigger_incremental_gc(
        &self,
        budget_pages: usize,
    ) -> Option<GCMetrics> {
        let infos = self.allocator.page_infos();
        let snapshot = self.build_snapshot(&infos);
        if snapshot.total_bytes <= self.config.gc_threshold_bytes
            && snapshot.cold_pages.is_empty()
        {
            return None;
        }
        let mut info_map = HashMap::new();
        for info in &infos {
            info_map.insert(info.page_id, info.size);
        }
        let target = if budget_pages == 0 {
            self.config.incremental_batch_pages
        } else {
            budget_pages.min(self.config.incremental_batch_pages.max(1))
        };
        let mut reclaimed_pages = 0usize;
        let mut reclaimed_bytes = 0usize;
        let start = Instant::now();
        for page_id in snapshot.cold_pages.into_iter().take(target) {
            if let Some(bytes) = info_map.get(&page_id).copied() {
                self.allocator.free(page_id);
                self.aging.lock().remove(&page_id);
                reclaimed_pages += 1;
                reclaimed_bytes += bytes;
            }
        }
        if reclaimed_pages == 0 {
            return None;
        }
        Some(GCMetrics {
            reclaimed_pages,
            reclaimed_bytes,
            duration: start.elapsed(),
        })
    }

    pub fn stats(&self) -> (u64, u64) {
        self.stats.snapshot()
    }

    pub fn config(&self) -> &MemoryMonitorConfig {
        &self.config
    }

    fn build_snapshot(&self, infos: &[crate::page::PageInfo]) -> MemorySnapshot {
        let total_pages = infos.len();
        let total_bytes: usize = infos.iter().map(|info| info.size).sum();
        let avg_bytes = if total_pages == 0 {
            0
        } else {
            total_bytes / total_pages
        };
        let cold_pages = self.update_aging(infos);
        MemorySnapshot {
            total_pages,
            total_bytes,
            avg_bytes_per_page: avg_bytes,
            cold_pages,
            collected_at: Instant::now(),
        }
    }

    fn update_aging(
        &self,
        infos: &[crate::page::PageInfo],
    ) -> Vec<PageID> {
        let mut aging = self.aging.lock();
        let mut cold = Vec::new();
        let mut seen = HashSet::new();
        for info in infos {
            seen.insert(info.page_id);
            let entry = aging
                .entry(info.page_id)
                .or_insert(PageAging {
                    last_epoch: info.epoch,
                    age: 0,
                });
            if entry.last_epoch == info.epoch {
                entry.age += 1;
            } else {
                entry.last_epoch = info.epoch;
                entry.age = 0;
            }
            if entry.age > self.config.cold_page_age_limit {
                cold.push(info.page_id);
            }
        }
        aging.retain(|page_id, _| seen.contains(page_id));
        cold
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryMonitor, MemoryMonitorConfig};
    use crate::page::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
    use crate::types::Epoch;
    use std::sync::Arc;
    use std::time::Duration;

    fn allocator() -> Arc<PageAllocator> {
        Arc::new(PageAllocator::new(PageAllocatorConfig::default()))
    }

    #[test]
    fn snapshot_identifies_cold_pages() {
        let allocator = allocator();
        allocator
            .allocate_raw(PageID(1), 1024, Some(PageLocation::Cpu))
            .unwrap();
        let config = MemoryMonitorConfig {
            gc_threshold_bytes: 1024,
            cold_page_age_limit: 1,
            incremental_batch_pages: 4,
        };
        let monitor = MemoryMonitor::with_config(Arc::clone(&allocator), config);
        let first = monitor.snapshot();
        assert!(first.cold_pages.is_empty());
        let second = monitor.snapshot();
        assert_eq!(second.cold_pages, vec![PageID(1)]);
        unsafe {
            let page = &mut *allocator.acquire_page(PageID(1)).unwrap();
            page.set_epoch(Epoch(42));
        }
        let third = monitor.snapshot();
        assert!(third.cold_pages.is_empty());
    }

    #[test]
    fn incremental_gc_reclaims_pages_under_budget() {
        let allocator = allocator();
        for id in 1..=4 {
            allocator
                .allocate_raw(PageID(id), 2048, Some(PageLocation::Cpu))
                .unwrap();
        }
        let config = MemoryMonitorConfig {
            gc_threshold_bytes: 2048,
            cold_page_age_limit: 0,
            incremental_batch_pages: 4,
        };
        let monitor = MemoryMonitor::with_config(Arc::clone(&allocator), config);
        let metrics = monitor.trigger_incremental_gc(4).unwrap();
        assert!(metrics.reclaimed_pages <= 4);
        assert!(metrics.duration < Duration::from_millis(3));
        assert!(monitor.snapshot().total_bytes <= monitor.config().gc_threshold_bytes);
    }

    #[test]
    fn snapshot_reflects_allocator_state() {
        let allocator = allocator();
        allocator
            .allocate_raw(PageID(1), 1024, Some(PageLocation::Cpu))
            .unwrap();
        allocator
            .allocate_raw(PageID(2), 2048, Some(PageLocation::Cpu))
            .unwrap();
        let monitor = MemoryMonitor::new(Arc::clone(&allocator));
        let snapshot = monitor.snapshot();
        assert_eq!(snapshot.total_pages, 2);
        assert_eq!(snapshot.total_bytes, 3072);
        assert_eq!(snapshot.avg_bytes_per_page, 1536);
    }

    #[test]
    fn gc_trigger_depends_on_threshold() {
        let allocator = allocator();
        allocator
            .allocate_raw(PageID(1), 4096, Some(PageLocation::Cpu))
            .unwrap();
        let config = MemoryMonitorConfig {
            gc_threshold_bytes: 1024,
            cold_page_age_limit: 0,
            incremental_batch_pages: 1,
        };
        let monitor = MemoryMonitor::with_config(Arc::clone(&allocator), config);
        assert!(monitor.trigger_incremental_gc(1).is_some());
        let monitor = MemoryMonitor::with_config(
            Arc::clone(&allocator),
            MemoryMonitorConfig {
                gc_threshold_bytes: 8192,
                cold_page_age_limit: 0,
                incremental_batch_pages: 1,
            },
        );
        assert!(monitor.trigger_incremental_gc(1).is_none());
    }
}
