use crate::delta::delta::Delta;
use crate::types::{PageID, Epoch};

#[derive(Debug, Clone)]
pub struct MaterializedPageView {
    pub page_id: PageID,
    pub epoch: Epoch,
    pub data: Vec<u8>,
    pub mask: Vec<bool>,
}
impl MaterializedPageView {
    pub fn empty(page_id: PageID, size: usize) -> Self {
        Self {
            page_id,
            epoch: Epoch(0),
            data: vec![0u8; size],
            mask: vec![false; size],
        }
    }
    pub fn size(&self) -> usize {
        self.data.len()
    pub fn read_byte(&self, offset: usize) -> Option<u8> {
        if offset < self.data.len() && self.mask[offset] {
            Some(self.data[offset])
        } else {
            None
pub fn materialize_page_state(
    page_id: PageID,
    deltas: &[Delta],
    page_size: usize,
) -> MaterializedPageView {
    let mut view = MaterializedPageView::empty(page_id, page_size);
    
    for delta in deltas {
        if delta.page_id != page_id {
            continue;
        
        view.epoch = view.epoch.max(delta.epoch);
        apply_delta_to_view(&mut view, delta);
    view
fn apply_delta_to_view(view: &mut MaterializedPageView, delta: &Delta) {
    let mut payload_idx = 0;
    for i in 0..view.size() {
        let changed = if i < delta.mask.len() {
            delta.mask[i]
            false
        };
        if changed {
            if delta.is_sparse {
                if payload_idx < delta.payload.len() {
                    view.data[i] = delta.payload[payload_idx];
                    payload_idx += 1;
                }
            } else if i < delta.payload.len() {
                view.data[i] = delta.payload[i];
            }
            
            view.mask[i] = true;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Delta, DeltaID, Source};
    #[test]
    fn materialize_empty_page() {
        let view = MaterializedPageView::empty(PageID(1), 10);
        assert_eq!(view.page_id, PageID(1));
        assert_eq!(view.size(), 10);
        assert_eq!(view.read_byte(0), None);
    fn materialize_single_delta() {
        let delta = Delta {
            delta_id: DeltaID(1),
            page_id: PageID(1),
            epoch: Epoch(1),
            mask: vec![true, false, true],
            payload: vec![0xAA, 0xBB, 0xCC],
            is_sparse: false,
            timestamp: 0,
            source: Source("test".to_string()),
            intent_metadata: None,
        let view = materialize_page_state(PageID(1), &[delta], 3);
        assert_eq!(view.epoch, Epoch(1));
        assert_eq!(view.read_byte(0), Some(0xAA));
        assert_eq!(view.read_byte(1), None);
        assert_eq!(view.read_byte(2), Some(0xCC));
    fn materialize_multiple_deltas() {
        let delta1 = Delta {
            mask: vec![true, false],
            payload: vec![0xAA, 0xBB],
        let delta2 = Delta {
            delta_id: DeltaID(2),
            epoch: Epoch(2),
            mask: vec![false, true],
            payload: vec![0xCC, 0xDD],
            timestamp: 1,
        let view = materialize_page_state(PageID(1), &[delta1, delta2], 2);
        assert_eq!(view.epoch, Epoch(2));
        assert_eq!(view.read_byte(1), Some(0xDD));
    fn materialize_sparse_delta() {
            mask: vec![true, false, true, false],
            is_sparse: true,
        let view = materialize_page_state(PageID(1), &[delta], 4);
        assert_eq!(view.read_byte(2), Some(0xBB));
        assert_eq!(view.read_byte(3), None);
