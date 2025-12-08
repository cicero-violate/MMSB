use crate::types::{Delta, DeltaError};

pub fn merge_deltas(first: &Delta, second: &Delta) -> Result<Delta, DeltaError> {
    first.merge(second)
}
