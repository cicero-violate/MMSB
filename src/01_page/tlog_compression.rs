use crate::page::Delta;

pub fn compact(deltas: &[Delta]) -> Vec<Delta> {
    if deltas.len() <= 1 {
        return deltas.to_vec();
    }
    let mut result = Vec::with_capacity(deltas.len());
    let mut iter = deltas.iter();
    if let Some(first) = iter.next() {
        result.push(first.clone());
        for delta in iter {
            if let Some(last) = result.last_mut() {
                if last.page_id == delta.page_id {
                    if let Ok(merged) = last.merge(delta) {
                        *last = merged;
                        continue;
                    }
                }
            }
            result.push(delta.clone());
        }
    }
    result
}
