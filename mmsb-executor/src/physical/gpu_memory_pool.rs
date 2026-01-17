use std::collections::HashMap;
use std::ffi::c_void;
use parking_lot::Mutex;

extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

const SLAB_SIZES: &[usize] = &[
    4096,      // 4KB
    16384,     // 16KB
    65536,     // 64KB
    262144,    // 256KB
    1048576,   // 1MB
    4194304,   // 4MB
];

#[derive(Debug)]
struct Slab {
    size: usize,
    free_blocks: Vec<*mut c_void>,
    allocated_count: usize,
}

#[derive(Debug)]
pub struct GPUMemoryPool {
    slabs: Mutex<HashMap<usize, Slab>>,
    stats: Mutex<PoolStats>,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub bytes_allocated: u64,
    pub bytes_cached: u64,
}

impl GPUMemoryPool {
    pub fn new() -> Self {
        let mut slabs = HashMap::new();
        for &size in SLAB_SIZES {
            slabs.insert(size, Slab {
                size,
                free_blocks: Vec::new(),
                allocated_count: 0,
            });
        }
        
        Self {
            slabs: Mutex::new(slabs),
            stats: Mutex::new(PoolStats::default()),
        }
    }
    
    fn select_slab_size(&self, size: usize) -> usize {
        SLAB_SIZES.iter()
            .find(|&&s| s >= size)
            .copied()
            .unwrap_or(*SLAB_SIZES.last().unwrap())
    }
    
    pub fn allocate(&self, size: usize) -> Result<*mut c_void, i32> {
        let slab_size = self.select_slab_size(size);
        let mut slabs = self.slabs.lock();
        let mut stats = self.stats.lock();
        
        stats.allocations += 1;
        
        if let Some(slab) = slabs.get_mut(&slab_size) {
            if let Some(ptr) = slab.free_blocks.pop() {
                stats.cache_hits += 1;
                slab.allocated_count += 1;
                return Ok(ptr);
            }
        }
        
        stats.cache_misses += 1;
        stats.bytes_allocated += slab_size as u64;
        
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { cudaMalloc(&mut ptr, slab_size) };
        
        if result == 0 {
            if let Some(slab) = slabs.get_mut(&slab_size) {
                slab.allocated_count += 1;
            }
            Ok(ptr)
        } else {
            Err(result)
        }
    }
    
    pub fn deallocate(&self, ptr: *mut c_void, size: usize) {
        let slab_size = self.select_slab_size(size);
        let mut slabs = self.slabs.lock();
        let mut stats = self.stats.lock();
        
        stats.deallocations += 1;
        
        if let Some(slab) = slabs.get_mut(&slab_size) {
            slab.free_blocks.push(ptr);
            slab.allocated_count -= 1;
            stats.bytes_cached += slab_size as u64;
        }
    }
    
    pub fn get_stats(&self) -> PoolStats {
        self.stats.lock().clone()
    }
    
    pub fn clear(&self) {
        let mut slabs = self.slabs.lock();
        for slab in slabs.values_mut() {
            for ptr in slab.free_blocks.drain(..) {
                unsafe { cudaFree(ptr) };
            }
        }
        
        let mut stats = self.stats.lock();
        stats.bytes_cached = 0;
    }
}

impl Drop for GPUMemoryPool {
    fn drop(&mut self) {
        self.clear();
    }
}
