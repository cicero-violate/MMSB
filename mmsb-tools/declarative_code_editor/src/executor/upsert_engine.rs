use crate::buffer::EditBuffer;
use crate::error::EditorError;
use crate::executor::mutation_planner::{plan_mutations, PlannedEdit};
use crate::upsert::{AnchorPosition, AnchorSpec, AnchorTarget, OnMissing, UpsertResult, UpsertSpec};
use crate::types::Edit;

/// Execute upsert with transactional semantics
pub fn execute_upsert(
    spec: &UpsertSpec,
    buffer: &mut EditBuffer,
) -> Result<UpsertResult, EditorError> {
    if buffer.has_edits() {
        return Err(EditorError::DirtyBuffer);
    }

    // Take snapshot for rollback
    let snapshot = buffer.clone();

    let result = execute_upsert_inner(spec, buffer);

    // Rollback on error
    if result.is_err() {
        *buffer = snapshot;
    }

    result
}

fn execute_upsert_inner(
    spec: &UpsertSpec,
    buffer: &mut EditBuffer,
) -> Result<UpsertResult, EditorError> {
    let allow_empty = spec.on_missing != OnMissing::Error;
    let planned = plan_mutations(&spec.plan, buffer, allow_empty, spec.allow_multiple)?;

    if spec.dry_run {
        return handle_dry_run(spec, buffer, &planned);
    }

    // Handle insert-on-miss
    if planned.is_empty() {
        return handle_missing(spec, buffer);
    }

    // Apply edits
    let before = buffer.render();
    for edit_plan in &planned {
        buffer.add_edit(edit_plan.edit.clone());
    }
    let after = buffer.render();

    let affected = planned.iter().map(|p| p.item_name.clone()).collect();
    let applied = before != after;
    let inserted = false;

    Ok(UpsertResult::new(applied, inserted).with_affected(affected))
}

fn handle_dry_run(
    spec: &UpsertSpec,
    buffer: &EditBuffer,
    planned: &[PlannedEdit],
) -> Result<UpsertResult, EditorError> {
    if planned.is_empty() && spec.on_missing == OnMissing::Insert {
        if let Some(anchor) = &spec.anchor {
            if let Some(value) = &spec.insert_value {
                let edit = resolve_anchor_edit(anchor, buffer, value)?;
                let before = buffer.source();
                let after = apply_edit_to_string(before, &edit);
                let diff = compute_diff(before, &after);
                
                return Ok(UpsertResult::new(false, true).with_diff(diff));
            }
        }
    }

    let before = buffer.render();
    let mut temp = buffer.clone();
    for edit_plan in planned {
        temp.add_edit(edit_plan.edit.clone());
    }
    let after = temp.render();
    let diff = compute_diff(&before, &after);

    let affected = planned.iter().map(|p| p.item_name.clone()).collect();
    Ok(UpsertResult::new(false, false).with_diff(diff).with_affected(affected))
}

fn handle_missing(
    spec: &UpsertSpec,
    buffer: &mut EditBuffer,
) -> Result<UpsertResult, EditorError> {
    match spec.on_missing {
        OnMissing::Error => Err(EditorError::NoMatches),
        OnMissing::Skip => Ok(UpsertResult::new(false, false)),
        OnMissing::Insert => {
            let anchor = spec.anchor.as_ref()
                .ok_or_else(|| EditorError::InvalidAnchor("No anchor specified".into()))?;
            let value = spec.insert_value.as_ref()
                .ok_or_else(|| EditorError::InvalidAnchor("No insert value specified".into()))?;

            let edit = resolve_anchor_edit(anchor, buffer, value)?;
            buffer.add_edit(edit);

            Ok(UpsertResult::new(true, true))
        }
    }
}

fn resolve_anchor_edit(
    anchor: &AnchorSpec,
    buffer: &EditBuffer,
    value: &str,
) -> Result<Edit, EditorError> {
    let source = buffer.source();
    let position = match &anchor.target {
        AnchorTarget::Top => 0u32,
        AnchorTarget::Bottom => source.len() as u32,
        AnchorTarget::Item { kind: _, name } => {
            // Find item by kind and name
            // Simplified: just search for name
            source.find(name)
                .map(|pos| pos as u32)
                .ok_or_else(|| EditorError::InvalidAnchor(format!("Item {} not found", name)))?
        }
        AnchorTarget::Query(_query) => {
            return Err(EditorError::UnsupportedOperation("Query-based anchors not yet implemented".into()));
        }
    };

    let (start, end) = match anchor.position {
        AnchorPosition::Before => (position, position),
        AnchorPosition::After => (position, position),
    };

    Ok(Edit::new(start, end, value))
}

fn apply_edit_to_string(source: &str, edit: &Edit) -> String {
    let mut result = source.to_string();
    let start = edit.start_byte as usize;
    let end = edit.old_end_byte as usize;
    result.replace_range(start..end, &edit.new_text);
    result
}

fn compute_diff(_before: &str, after: &str) -> String {
    // Simple diff - in production would use a proper diff library
    format!("--- before\n+++ after\n{}\n", after)
}
