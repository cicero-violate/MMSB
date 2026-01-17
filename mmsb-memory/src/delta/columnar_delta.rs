//! Columnar delta batch representation optimized for SIMD scans.

use crate::delta::delta::Delta;
use crate::types::page::Page;
use crate::types::{DeltaID, Epoch, PageError, PageID, Source};
use std::collections::HashMap;
#[derive(Debug, Clone)]
pub struct ColumnarDeltaBatch {
    count: usize,
    delta_ids: Vec<DeltaID>,
    page_ids: Vec<PageID>,
    epochs: Vec<Epoch>,
    is_sparse: Vec<bool>,
    mask_offsets: Vec<(usize, usize)>,
    payload_offsets: Vec<(usize, usize)>,
    mask_pool: Vec<u8>,
    payload_pool: Vec<u8>,
    timestamps: Vec<u64>,
    sources: Vec<Source>,
    metadata: Vec<Option<String>>,
}
impl ColumnarDeltaBatch {
    pub fn new() -> Self {
        Self {
            count: 0,
            delta_ids: Vec::new(),
            page_ids: Vec::new(),
            epochs: Vec::new(),
            is_sparse: Vec::new(),
            mask_offsets: Vec::new(),
            payload_offsets: Vec::new(),
            mask_pool: Vec::new(),
            payload_pool: Vec::new(),
            timestamps: Vec::new(),
            sources: Vec::new(),
            metadata: Vec::new(),
        }
    }
    pub fn from_rows(deltas: Vec<Delta>) -> Self {
        let mut batch = Self::new();
        batch.extend(deltas);
        batch
    pub fn len(&self) -> usize {
        self.count
    pub fn is_empty(&self) -> bool {
        self.count == 0
    pub fn iter(&self) -> impl Iterator<Item = Delta> + '_ {
        (0..self.count).map(move |idx| self.delta_at(idx).expect("index in range"))
    pub fn extend<I>(&mut self, deltas: I)
    where
        I: IntoIterator<Item = Delta>,
    {
        for delta in deltas {
            let mask_start = self.mask_pool.len();
            self.mask_pool
                .extend(delta.mask.iter().map(|flag| if *flag { 1 } else { 0 }));
            self.mask_offsets.push((mask_start, delta.mask.len()));
            let payload_start = self.payload_pool.len();
            self.payload_pool.extend(&delta.payload);
            self.payload_offsets
                .push((payload_start, delta.payload.len()));
            self.delta_ids.push(delta.delta_id);
            self.page_ids.push(delta.page_id);
            self.epochs.push(delta.epoch);
            self.is_sparse.push(delta.is_sparse);
            self.timestamps.push(delta.timestamp);
            self.sources.push(delta.source);
            self.metadata.push(delta.intent_metadata);
            self.count += 1;
    pub fn to_vec(&self) -> Vec<Delta> {
        self.iter().collect()
    pub fn filter_epoch_eq(&self, epoch: Epoch) -> Vec<usize> {
        let mut matches = Vec::new();
        let words = self.epoch_words();
        let mut idx = 0usize;
        const LANES: usize = 8;
        while idx + LANES <= words.len() {
            let mut lane_mask = 0u8;
            for lane in 0..LANES {
                if words[idx + lane] == epoch.0 {
                    lane_mask |= 1 << lane;
                }
            }
            if lane_mask != 0 {
                for lane in 0..LANES {
                    if (lane_mask >> lane) & 1 == 1 {
                        matches.push(idx + lane);
                    }
            idx += LANES;
        for (offset, value) in words[idx..].iter().enumerate() {
            if *value == epoch.0 {
                matches.push(idx + offset);
        matches
    pub fn scan_page_id(&self, target: PageID) -> Vec<usize> {
        let ids = self.page_words();
        const LANES: usize = 4;
        while idx + LANES <= ids.len() {
                if ids[idx + lane] == target.0 {
        for (offset, value) in ids[idx..].iter().enumerate() {
            if *value == target.0 {
    pub fn apply_to_pages(
        &self,
        pages: &mut HashMap<PageID, Page>,
    ) -> Result<(), PageError> {
        for idx in 0..self.count {
            let delta = self.delta_at(idx).expect("index valid");
            if let Some(page) = pages.get_mut(&delta.page_id) {
                page.apply_delta(&delta)?;
            } else {
                return Err(PageError::PageNotFound(delta.page_id));
        Ok(())
    pub fn delta_at(&self, idx: usize) -> Option<Delta> {
        (idx < self.count).then(|| self.build_delta(idx))
    pub fn page_id_at(&self, idx: usize) -> Option<PageID> {
        self.page_ids.get(idx).copied()
    fn build_delta(&self, idx: usize) -> Delta {
        let (mask_start, mask_len) = self.mask_offsets[idx];
        let mask = self.mask_pool[mask_start..mask_start + mask_len]
            .iter()
            .map(|b| *b != 0)
            .collect::<Vec<bool>>();
        let (payload_start, payload_len) = self.payload_offsets[idx];
        let payload = self.payload_pool[payload_start..payload_start + payload_len].to_vec();
        Delta {
            delta_id: self.delta_ids[idx],
            page_id: self.page_ids[idx],
            epoch: self.epochs[idx],
            mask,
            payload,
            is_sparse: self.is_sparse[idx],
            timestamp: self.timestamps[idx],
            source: self.sources[idx].clone(),
            intent_metadata: self.metadata[idx].clone(),
    fn epoch_words(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.epochs.as_ptr() as *const u32, self.epochs.len()) }
    fn page_words(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.page_ids.as_ptr() as *const u64, self.page_ids.len()) }
impl Default for ColumnarDeltaBatch {
    fn default() -> Self {
        Self::new()
impl From<Vec<Delta>> for ColumnarDeltaBatch {
    fn from(value: Vec<Delta>) -> Self {
        Self::from_rows(value)
#[cfg(test)]
mod tests {
    use super::ColumnarDeltaBatch;
    use crate::types::{Delta, DeltaID, Page, PageID, PageLocation, Source};
    use crate::types::Epoch;
    use std::collections::HashMap;
    fn make_delta(id: u64, page: u64, epoch: u32, payload: &[u8]) -> Delta {
            delta_id: DeltaID(id),
            page_id: PageID(page),
            epoch: Epoch(epoch),
            mask: payload.iter().map(|_| true).collect(),
            payload: payload.to_vec(),
            is_sparse: false,
            timestamp: id,
            source: Source(format!("src-{id}")),
            intent_metadata: None,
    #[test]
    fn test_roundtrip() {
        let deltas = vec![
            make_delta(1, 10, 5, b"abc"),
            make_delta(2, 10, 6, b"def"),
        ];
        let batch = ColumnarDeltaBatch::from_rows(deltas.clone());
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.to_vec().len(), 2);
        let back = batch.to_vec();
        assert_eq!(back[0].delta_id.0, 1);
        assert_eq!(back[1].payload, b"def");
    fn test_epoch_filter() {
            make_delta(1, 1, 1, b"a"),
            make_delta(2, 2, 2, b"b"),
            make_delta(3, 3, 1, b"c"),
        let batch = ColumnarDeltaBatch::from_rows(deltas);
        let matches = batch.filter_epoch_eq(Epoch(1));
        assert_eq!(matches, vec![0, 2]);
    fn test_apply_to_pages() {
        let deltas = vec![make_delta(1, 1, 1, b"\x01\x02"), make_delta(2, 2, 2, b"\xFF\xEE")];
        let mut pages = HashMap::new();
        let mut page1 = Page::new(PageID(1), 2, PageLocation::Cpu).unwrap();
        let mut page2 = Page::new(PageID(2), 2, PageLocation::Cpu).unwrap();
        pages.insert(PageID(1), page1);
        pages.insert(PageID(2), page2);
        batch.apply_to_pages(&mut pages).unwrap();
        page1 = pages.remove(&PageID(1)).unwrap();
        page2 = pages.remove(&PageID(2)).unwrap();
        assert_eq!(page1.data_slice(), b"\x01\x02");
        assert_eq!(page2.data_slice(), b"\xFF\xEE");
