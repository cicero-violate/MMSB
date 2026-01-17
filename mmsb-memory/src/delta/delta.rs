use crate::types::page::Page;
use crate::types::{DeltaError, DeltaID, Epoch, PageError, PageID, Source};
use std::time::{SystemTime, UNIX_EPOCH};

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
    pub intent_metadata: Option<String>,
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
            intent_metadata: None,
        })
    }
    pub fn new_sparse(
        payload: Vec<u8>,
        let changed = mask.iter().filter(|&&m| m).count();
        if changed != payload.len() {
                payload_len: payload.len(),
            payload,
            is_sparse: true,
    pub fn merge(&self, other: &Delta) -> Result<Delta, DeltaError> {
        if self.page_id != other.page_id {
            return Err(DeltaError::PageIDMismatch {
                expected: self.page_id,
                found: other.page_id,
        if self.mask.len() != other.mask.len() {
            return Err(DeltaError::MaskSizeMismatch {
                expected: self.mask.len(),
                found: other.mask.len(),
        let mut merged_mask = self.mask.clone();
        let mut merged_payload = self.to_dense();
        let other_dense = other.to_dense();
        for (idx, &flag) in other.mask.iter().enumerate() {
            if flag {
                merged_mask[idx] = true;
                merged_payload[idx] = other_dense[idx];
            }
        Ok(Delta {
            delta_id: other.delta_id,
            page_id: other.page_id,
            epoch: Epoch(other.epoch.0.max(self.epoch.0)),
            mask: merged_mask,
            payload: merged_payload,
            timestamp: other.timestamp.max(self.timestamp),
            source: other.source.clone(),
            intent_metadata: other.intent_metadata.clone().or_else(|| self.intent_metadata.clone()),
    pub fn to_dense(&self) -> Vec<u8> {
        if !self.is_sparse {
            return self.payload.clone();
        let mut dense = vec![0u8; self.mask.len()];
        let mut payload_idx = 0;
        for (idx, &flag) in self.mask.iter().enumerate() {
                dense[idx] = self.payload[payload_idx];
                payload_idx += 1;
        dense
    pub fn apply_to(&self, page: &mut Page) -> Result<(), PageError> {
        if let Err(err) = crate::delta::delta_validation::validate_delta(self) {
            return Err(match err {
                DeltaError::SizeMismatch { mask_len, payload_len } => PageError::MaskSizeMismatch {
                    expected: mask_len,
                    found: payload_len,
                },
                DeltaError::PageIDMismatch { expected, found } => PageError::PageIDMismatch {
                    expected,
                    found,
                DeltaError::MaskSizeMismatch { expected, found } => PageError::MaskSizeMismatch {
        page.apply_delta(self)
fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
