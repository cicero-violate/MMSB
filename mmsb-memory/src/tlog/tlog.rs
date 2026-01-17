use crate::delta::delta::Delta;
use crate::types::{DeltaID, Epoch, PageID, Source};
use mmsb_proof::OutcomeProof;
use std::sync::RwLock;

pub struct TransactionLog {
    deltas: RwLock<Vec<Delta>>,
}
impl TransactionLog {
    pub fn new() -> Self {
        Self {
            deltas: RwLock::new(Vec::new()),
        }
    }
    
    pub fn append(&self, _proof: &ExecutionProof, delta: Delta) -> std::io::Result<()> {
        self.deltas.write().unwrap().push(delta);
        Ok(())
pub fn delta_hash(_delta: &Delta) -> [u8; 32] {
    [0u8; 32] // TODO(executor): hashing moved upstream
