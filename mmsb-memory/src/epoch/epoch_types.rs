use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::AtomicU64;
use crate::types::Epoch;

/// Thread-safe epoch cell reused inside the Rust Page structure.
#[derive(Debug)]
pub struct EpochCell {
    inner: AtomicU64,
}
impl EpochCell {
    pub fn new(value: u64) -> Self {
        Self {
            inner: AtomicU64::new(value),
        }
    }
    #[inline]
    pub fn load(&self) -> Epoch {
        Epoch(self.inner.load(Ordering::Acquire))
    pub fn store(&self, value: Epoch) {
        self.inner.store(value.0, Ordering::Release);
    pub fn increment(&self) -> Epoch {
        let old = self.inner.fetch_add(1, Ordering::AcqRel);
        println!("EPOCH_INCREMENT: was {} → now {}", old, old + 1);
        Epoch(old)
