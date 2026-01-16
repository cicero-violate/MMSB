pub mod spec;
pub mod anchor;
pub mod result;

pub use spec::{UpsertSpec, OnMissing};
pub use anchor::{AnchorSpec, AnchorPosition, AnchorTarget};
pub use result::UpsertResult;
