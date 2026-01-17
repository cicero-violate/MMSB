use crate::delta::{ColumnarDeltaBatch, Delta, merge_deltas, DeltaError};
use crate::page::{PageAllocator, PageError, PageID};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub struct ThroughputEngine {
    allocator: Arc<PageAllocator>,
    pool: ThreadPool,
    batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub processed: usize,
    pub duration: Duration,
    pub throughput: f64,
    pub batches: usize,
}

impl ThroughputMetrics {
    fn new(processed: usize, duration: Duration, batch_size: usize) -> Self {
        let throughput = if duration.as_secs_f64() == 0.0 {
            processed as f64
        } else {
            processed as f64 / duration.as_secs_f64()
        };
        let batches = if batch_size == 0 {
            0
        } else {
            (processed + batch_size - 1) / batch_size
        };
        Self {
            processed,
            duration,
            throughput,
            batches,
        }
    }
}

impl ThroughputEngine {
    pub fn new(
        allocator: Arc<PageAllocator>,
        workers: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            allocator,
            pool: ThreadPool::new(workers.max(1)),
            batch_size: batch_size.max(1),
        }
    }

    pub fn process_parallel(
        &self,
        deltas: Vec<Delta>,
    ) -> Result<ThroughputMetrics, PageError> {
        let start = Instant::now();
        if deltas.is_empty() {
            return Ok(ThroughputMetrics::new(0, Duration::default(), self.batch_size));
        }
        let batch = Arc::new(ColumnarDeltaBatch::from_rows(deltas));
        let partitions = partition_by_page(&batch);
        if partitions.is_empty() {
            return Ok(ThroughputMetrics::new(0, Duration::default(), self.batch_size));
        }

        let chunks = chunk_partitions(partitions, self.pool.worker_count());
        let (result_tx, result_rx) = mpsc::channel();
        for chunk in chunks {
            let allocator = Arc::clone(&self.allocator);
            let columnar = Arc::clone(&batch);
            let tx = result_tx.clone();
            self.pool.execute(move || {
                let result = process_chunk(chunk, allocator, columnar);
                tx.send(result).ok();
            });
        }
        drop(result_tx);

        let mut processed = 0usize;
        for result in result_rx {
            match result {
                Ok(count) => processed += count,
                Err(err) => return Err(err),
            }
        }
        Ok(ThroughputMetrics::new(processed, start.elapsed(), self.batch_size))
    }
}

fn partition_by_page(batch: &ColumnarDeltaBatch) -> Vec<(PageID, Vec<usize>)> {
    let mut map: HashMap<PageID, Vec<usize>> = HashMap::new();
    for idx in 0..batch.len() {
        if let Some(page_id) = batch.page_id_at(idx) {
            map.entry(page_id).or_default().push(idx);
        }
    }
    map.into_iter().collect()
}

fn chunk_partitions(
    partitions: Vec<(PageID, Vec<usize>)>,
    workers: usize,
) -> Vec<Vec<(PageID, Vec<usize>)>> {
    if partitions.is_empty() {
        return Vec::new();
    }
    let chunk_size = ((partitions.len() + workers - 1) / workers).max(1);
    let mut chunks = Vec::new();
    let mut current = Vec::with_capacity(chunk_size);
    for entry in partitions {
        current.push(entry);
        if current.len() == chunk_size {
            chunks.push(current);
            current = Vec::with_capacity(chunk_size);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn process_chunk(
    chunk: Vec<(PageID, Vec<usize>)>,
    allocator: Arc<PageAllocator>,
    batch: Arc<ColumnarDeltaBatch>,
) -> Result<usize, PageError> {
    let mut processed = 0usize;
    for (page_id, indexes) in chunk {
        let ptr = allocator
            .acquire_page(page_id)
            .ok_or(PageError::PageNotFound(page_id))?;
        if indexes.is_empty() {
            continue;
        }
        let mut merged: Option<Delta> = None;
        for idx in indexes {
            if let Some(delta) = batch.delta_at(idx) {
                processed += 1;
                merged = Some(match merged {
                    Some(ref current) => merge_deltas(current, &delta)
                        .map_err(delta_error_to_page)?,
                    None => delta,
                });
            }
        }
        if let Some(final_delta) = merged {
            unsafe {
                (*ptr).apply_delta(&final_delta)?;
            }
        }
    }
    Ok(processed)
}

fn delta_error_to_page(err: DeltaError) -> PageError {
    match err {
        DeltaError::SizeMismatch { mask_len, payload_len } => {
            PageError::MaskSizeMismatch {
                expected: mask_len,
                found: payload_len,
            }
        }
        DeltaError::PageIDMismatch { expected, found } => PageError::PageIDMismatch {
            expected,
            found,
        },
        DeltaError::MaskSizeMismatch { expected, found } => PageError::MaskSizeMismatch {
            expected,
            found,
        },
    }
}

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    Job(Job),
    Shutdown,
}

struct ThreadPool {
    sender: mpsc::Sender<Message>,
    workers: usize,
    handles: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<Message>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut handles = Vec::with_capacity(size);
        for _ in 0..size {
            let rx = Arc::clone(&receiver);
            handles.push(thread::spawn(move || loop {
                let message = rx.lock().expect("receiver poisoned").recv();
                match message {
                    Ok(Message::Job(job)) => job(),
                    Ok(Message::Shutdown) | Err(_) => break,
                }
            }));
        }
        Self {
            sender,
            workers: size,
            handles,
        }
    }

    fn execute<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let _ = self.sender.send(Message::Job(Box::new(job)));
    }

    fn worker_count(&self) -> usize {
        self.workers
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in 0..self.workers {
            let _ = self.sender.send(Message::Shutdown);
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{DeltaID, PageAllocator, PageID, PageLocation, Source};
    use crate::epoch::Epoch;

    fn make_delta(id: u64, page: u64, payload: &[u8]) -> Delta {
        Delta {
            delta_id: DeltaID(id),
            page_id: PageID(page),
            epoch: Epoch(id as u32),
            mask: payload.iter().map(|_| true).collect(),
            payload: payload.to_vec(),
            is_sparse: false,
            timestamp: id,
            source: Source(format!("src-{id}")),
            intent_metadata: None,
        }
    }

    #[test]
    fn applies_batches_in_parallel() {
        let allocator = Arc::new(PageAllocator::new(Default::default()));
        allocator
            .allocate_raw(PageID(1), 4, Some(PageLocation::Cpu))
            .unwrap();
        allocator
            .allocate_raw(PageID(2), 4, Some(PageLocation::Cpu))
            .unwrap();
        let engine = ThroughputEngine::new(Arc::clone(&allocator), 2, 4);
        let deltas = vec![
            make_delta(1, 1, b"\x01\x02\x03\x04"),
            make_delta(2, 2, b"\xAA\xBB\xCC\xDD"),
        ];
        let metrics = engine.process_parallel(deltas).unwrap();
        assert_eq!(metrics.processed, 2);
        unsafe {
            let page1 = &mut *allocator.acquire_page(PageID(1)).unwrap();
            let page2 = &mut *allocator.acquire_page(PageID(2)).unwrap();
            assert_eq!(page1.data_slice(), b"\x01\x02\x03\x04");
            assert_eq!(page2.data_slice(), b"\xAA\xBB\xCC\xDD");
        }
    }

    #[test]
    fn merges_multiple_deltas_per_page() {
        let allocator = Arc::new(PageAllocator::new(Default::default()));
        allocator
            .allocate_raw(PageID(1), 4, Some(PageLocation::Cpu))
            .unwrap();
        let engine = ThroughputEngine::new(Arc::clone(&allocator), 1, 4);
        let deltas = vec![
            make_delta(1, 1, b"\x01\x02\x03\x04"),
            make_delta(2, 1, b"\x10\x20\x30\x40"),
        ];
        let metrics = engine.process_parallel(deltas).unwrap();
        assert_eq!(metrics.processed, 2);
        unsafe {
            let page1 = &mut *allocator.acquire_page(PageID(1)).unwrap();
            assert_eq!(page1.data_slice(), b"\x10\x20\x30\x40");
        }
    }

    #[test]
    fn reports_nonzero_throughput_for_large_batches() {
        let allocator = Arc::new(PageAllocator::new(Default::default()));
        for id in 1..=8 {
            allocator
                .allocate_raw(PageID(id), 8, Some(PageLocation::Cpu))
                .unwrap();
        }
        let engine = ThroughputEngine::new(Arc::clone(&allocator), 4, 256);
        let mut deltas = Vec::new();
        for i in 0..2000u64 {
            let page = 1 + (i % 8);
            deltas.push(make_delta(i + 1, page, &[i as u8; 8]));
        }
        let metrics = engine.process_parallel(deltas).unwrap();
        assert_eq!(metrics.processed, 2000);
        assert!(metrics.throughput > 0.0);
        assert!(metrics.batches >= 1);
    }
}
