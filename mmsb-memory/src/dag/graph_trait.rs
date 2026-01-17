use crate::types::PageID;
use crate::types::EdgeType;

pub trait GraphStructure {
    fn get_edges(&self) -> Vec<(PageID, PageID, EdgeType)>;
    fn get_children(&self, node: PageID) -> Vec<(PageID, EdgeType)>;
}
