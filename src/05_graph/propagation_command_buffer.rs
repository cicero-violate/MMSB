use crate::types::{Page, PageID};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PropagationCommand {
    pub page_id: PageID,
    pub page: Arc<Page>,
    pub dependencies: Vec<Arc<Page>>,
}
