use super::epoch::Epoch;
use super::page::{Page, PageError, PageID};
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeltaID(pub u64);

#[derive(Debug, Clone)]
pub struct Source(pub String);

#[derive(Debug, Clone)]
pub struct Delta {
    pub delta_id: DeltaID,
    pub page_id: PageID,
    pub epoch: Epoch,
    pub mask: Vec<bool>,
    pub payload: Vec<u8>,
    pub is_sparse: bool,
    pub timestamp: u64,
    pub source: Source,
}

impl Delta {
    pub fn new_dense(
        delta_id: DeltaID,
        page_id: PageID,
        epoch: Epoch,
        data: Vec<u8>,
        mask: Vec<bool>,
        source: Source,
    ) -> Result<Self, DeltaError> {
        if data.len() != mask.len() {
            return Err(DeltaError::SizeMismatch {
                mask_len: mask.len(),
                payload_len: data.len(),
            });
        }

        Ok(Self {
            delta_id,
            page_id,
            epoch,
            mask,
            payload: data,
            is_sparse: false,
            timestamp: now_ns(),
            source,
        })
    }

    pub fn new_sparse(
        delta_id: DeltaID,
        page_id: PageID,
        epoch: Epoch,
        mask: Vec<bool>,
        payload: Vec<u8>,
        source: Source,
    ) -> Result<Self, DeltaError> {
        let changed = mask.iter().filter(|&&m| m).count();
        if changed != payload.len() {
            return Err(DeltaError::SizeMismatch {
                mask_len: mask.len(),
                payload_len: payload.len(),
            });
        }

        Ok(Self {
            delta_id,
            page_id,
            epoch,
            mask,
            payload,
            is_sparse: true,
            timestamp: now_ns(),
            source,
        })
    }

    pub fn merge(&self, other: &Delta) -> Result<Delta, DeltaError> {
        if self.page_id != other.page_id {
            return Err(DeltaError::PageIDMismatch {
                expected: self.page_id,
                found: other.page_id,
            });
        }

        if self.mask.len() != other.mask.len() {
            return Err(DeltaError::MaskSizeMismatch {
                expected: self.mask.len(),
                found: other.mask.len(),
            });
        }

        let mut merged_mask = self.mask.clone();
        let mut merged_payload = self.to_dense();
        let other_dense = other.to_dense();

        for (idx, &flag) in other.mask.iter().enumerate() {
            if flag {
                merged_mask[idx] = true;
                merged_payload[idx] = other_dense[idx];
            }
        }

        Ok(Delta {
            delta_id: other.delta_id,
            page_id: other.page_id,
            epoch: Epoch(other.epoch.0.max(self.epoch.0)),
            mask: merged_mask,
            payload: merged_payload,
            is_sparse: false,
            timestamp: other.timestamp.max(self.timestamp),
            source: other.source.clone(),
        })
    }

    pub fn to_dense(&self) -> Vec<u8> {
        if !self.is_sparse {
            return self.payload.clone();
        }

        let mut dense = vec![0u8; self.mask.len()];
        let mut payload_idx = 0;
        for (idx, &flag) in self.mask.iter().enumerate() {
            if flag {
                dense[idx] = self.payload[payload_idx];
                payload_idx += 1;
            }
        }
        dense
    }

    pub fn apply_to(&self, page: &mut Page) -> Result<(), PageError> {
        page.apply_delta(self)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeltaError {
    #[error("Mask/payload size mismatch mask={mask_len} payload={payload_len}")]
    SizeMismatch { mask_len: usize, payload_len: usize },

    #[error("PageID mismatch: expected {expected:?}, found {found:?}")]
    PageIDMismatch { expected: PageID, found: PageID },

    #[error("Mask size mismatch: expected {expected}, found {found}")]
    MaskSizeMismatch { expected: usize, found: usize },
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
