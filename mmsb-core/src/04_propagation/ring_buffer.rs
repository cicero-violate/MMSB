//! Lock-free ring buffer optimized for propagation command dispatch.
//!
//! The implementation is based on a bounded MPMC queue where each slot tracks its
//! own sequence number. Atomic head/tail indices provide overall progress while
//! per-slot CAS operations guarantee correctness without locks.

use std::cell::UnsafeCell;
use std::iter::FromIterator;
use std::mem::MaybeUninit;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(align(64))]
struct CacheLine<T>(T);

impl<T> CacheLine<T> {
    fn new(value: T) -> Self {
        CacheLine(value)
    }
}

impl<T> std::ops::Deref for CacheLine<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for CacheLine<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

struct Slot<T> {
    sequence: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Slot<T> {
    fn new(sequence: usize) -> Self {
        Self {
            sequence: AtomicUsize::new(sequence),
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

/// Lock-free ring buffer with cache-friendly layout.
pub struct LockFreeRingBuffer<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    capacity: usize,
    head: CacheLine<AtomicUsize>,
    tail: CacheLine<AtomicUsize>,
}

unsafe impl<T: Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for LockFreeRingBuffer<T> {}

impl<T> LockFreeRingBuffer<T> {
    /// Create a new ring buffer with the requested capacity (rounded up to the next power of two).
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("capacity must be > 0").get().next_power_of_two();
        let slots = (0..cap)
            .map(|seq| Slot::new(seq))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            mask: cap - 1,
            capacity: cap,
            buffer: slots,
            head: CacheLine::new(AtomicUsize::new(0)),
            tail: CacheLine::new(AtomicUsize::new(0)),
        }
    }

    /// Current capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate item count.
    pub fn len(&self) -> usize {
        self.tail.load(Ordering::Acquire) - self.head.load(Ordering::Acquire)
    }

    /// Try to enqueue a single element.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        loop {
            let slot = unsafe { self.buffer.get_unchecked(pos & self.mask) };
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - pos as isize;
            if diff == 0 {
                match self
                    .tail
                    .compare_exchange_weak(pos, pos + 1, Ordering::AcqRel, Ordering::Relaxed)
                {
                    Ok(_) => {
                        unsafe { (*slot.value.get()).write(value); }
                        slot.sequence.store(pos + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                return Err(value);
            } else {
                pos = self.tail.load(Ordering::Relaxed);
            }
        }
    }

    /// Try to dequeue a single element.
    pub fn try_pop(&self) -> Option<T> {
        let mut pos = self.head.load(Ordering::Relaxed);
        loop {
            let slot = unsafe { self.buffer.get_unchecked(pos & self.mask) };
            let seq = slot.sequence.load(Ordering::Acquire);
            let diff = seq as isize - (pos + 1) as isize;
            if diff == 0 {
                match self
                    .head
                    .compare_exchange_weak(pos, pos + 1, Ordering::AcqRel, Ordering::Relaxed)
                {
                    Ok(_) => {
                        let value = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.sequence
                            .store(pos + self.capacity, Ordering::Release);
                        return Some(value);
                    }
                    Err(actual) => pos = actual,
                }
            } else if diff < 0 {
                return None;
            } else {
                pos = self.head.load(Ordering::Relaxed);
            }
        }
    }

    /// Push as many items from the iterator as possible, returning how many were accepted.
    pub fn push_batch<I>(&self, iter: I) -> usize
    where
        I: IntoIterator<Item = T>,
    {
        let mut written = 0usize;
        for value in iter {
            if self.try_push(value).is_err() {
                break;
            }
            written += 1;
        }
        written
    }

    /// Pop up to `max` elements and return them in FIFO order.
    pub fn pop_batch(&self, max: usize) -> Vec<T> {
        let mut drained = Vec::with_capacity(max);
        for _ in 0..max {
            match self.try_pop() {
                Some(value) => drained.push(value),
                None => break,
            }
        }
        drained
    }
}

impl<T> Drop for LockFreeRingBuffer<T> {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

impl<T> FromIterator<T> for LockFreeRingBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let items: Vec<T> = iter.into_iter().collect();
        let buffer = LockFreeRingBuffer::new(items.len().max(1));
        buffer.push_batch(items);
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::LockFreeRingBuffer;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_push_pop() {
        let buffer = LockFreeRingBuffer::new(4);
        assert!(buffer.is_empty());
        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        assert_eq!(buffer.try_pop(), Some(1));
        assert_eq!(buffer.try_pop(), Some(2));
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_wraparound_behavior() {
        let buffer = LockFreeRingBuffer::new(2);
        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();
        assert!(buffer.try_push(3).is_err());
        assert_eq!(buffer.try_pop(), Some(1));
        assert!(buffer.try_push(3).is_ok());
        assert_eq!(buffer.pop_batch(3), vec![2, 3]);
    }

    #[test]
    fn test_concurrent_producers_consumers() {
        let buffer = Arc::new(LockFreeRingBuffer::new(128));
        let total = 10_000;
        let produced = Arc::new(AtomicUsize::new(0));
        let consumed = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let buf = Arc::clone(&buffer);
            let counter = Arc::clone(&produced);
            handles.push(thread::spawn(move || {
                loop {
                    let next = counter.fetch_add(1, Ordering::SeqCst);
                    if next >= total {
                        break;
                    }
                    while buf.try_push(next).is_err() {
                        thread::yield_now();
                    }
                }
            }));
        }

        for _ in 0..2 {
            let buf = Arc::clone(&buffer);
            let counter = Arc::clone(&consumed);
            handles.push(thread::spawn(move || {
                while counter.load(Ordering::SeqCst) < total {
                    if buf.try_pop().is_some() {
                        counter.fetch_add(1, Ordering::SeqCst);
                    } else {
                        thread::sleep(Duration::from_micros(50));
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::SeqCst), total);
        assert!(buffer.is_empty());
    }
}
