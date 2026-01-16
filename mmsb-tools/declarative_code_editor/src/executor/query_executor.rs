use crate::buffer::EditBuffer;
use crate::error::EditorError;
use crate::query::QueryPlan;
use syn::Item;

/// Execute query and return matched items
pub fn execute_query<'a>(
    plan: &'a QueryPlan,
    buffer: &'a EditBuffer,
    allow_empty: bool,
    allow_multiple: bool,
) -> Result<Vec<&'a Item>, EditorError> {
    let matches = plan.execute(buffer);

    if matches.is_empty() && !allow_empty {
        return Err(EditorError::NoMatches);
    }

    if matches.len() > 1 && !allow_multiple {
        return Err(EditorError::MultipleMatches(matches.len()));
    }

    Ok(matches)
}
