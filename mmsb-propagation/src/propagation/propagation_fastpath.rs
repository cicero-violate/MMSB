use crate::page::Page;

/// Placeholder fast-path propagation implementation.
pub fn passthrough(_source: &Page, _target: &mut Page) {
    // Full implementation will stream bytes directly between pages once the
    // device manager and propagation queue own the backing storage.
}
