use crate::page::{Delta, Page};

pub fn apply_log(pages: &mut [Page], deltas: &[Delta]) {
    for delta in deltas {
        if let Some(page) = pages.iter_mut().find(|p| p.id == delta.page_id) {
            let _ = page.apply_delta(delta);
        }
    }
}
