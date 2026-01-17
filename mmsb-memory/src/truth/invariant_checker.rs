use crate::dag::DependencyGraph;
use crate::outcome::DagValidator;
use crate::device::DeviceRegistry;
use crate::page::{PageAllocator, PageError, PageID};
use crate::epoch::Epoch;
use parking_lot::RwLock;
use std::collections::HashMap;

pub trait Invariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn check(&self, ctx: &InvariantContext) -> InvariantResult;
}

#[derive(Clone)]
pub struct InvariantContext<'a> {
    pub allocator: Option<&'a PageAllocator>,
    pub graph: Option<&'a DependencyGraph>,
    pub registry: Option<&'a DeviceRegistry>,
}

#[derive(Debug, Clone)]
pub struct InvariantResult {
    pub name: &'static str,
    pub passed: bool,
    pub details: Option<String>,
}

impl InvariantResult {
    pub fn ok(name: &'static str) -> Self {
        Self {
            name,
            passed: true,
            details: None,
        }
    }

    pub fn fail(name: &'static str, msg: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            details: Some(msg.into()),
        }
    }
}

pub struct InvariantChecker {
    invariants: Vec<Box<dyn Invariant>>,
}

impl InvariantChecker {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
        }
    }

    pub fn with_builtins() -> Self {
        let mut checker = Self::new();
        checker.register(EpochMonotonicity::default());
        checker.register(PageConsistency::default());
        checker.register(GraphAcyclicity::new());
        checker
    }

    pub fn register<I>(&mut self, invariant: I)
    where
        I: Invariant + 'static,
    {
        self.invariants.push(Box::new(invariant));
    }

    pub fn run(&self, ctx: &InvariantContext<'_>) -> Vec<InvariantResult> {
        self.invariants.iter().map(|inv| inv.check(ctx)).collect()
    }
}

#[derive(Default)]
pub struct EpochMonotonicity {
    seen: RwLock<HashMap<PageID, Epoch>>,
}

impl Invariant for EpochMonotonicity {
    fn name(&self) -> &'static str {
        "EpochMonotonicity"
    }

    fn check(&self, ctx: &InvariantContext) -> InvariantResult {
        let allocator = match ctx.allocator {
            Some(alloc) => alloc,
            None => {
                return InvariantResult::fail(self.name(), "allocator unavailable");
            }
        };
        let mut guard = self.seen.write();
        for page in allocator.page_infos() {
            match guard.get(&page.page_id) {
                Some(epoch) if page.epoch < epoch.0 => {
                    return InvariantResult::fail(
                        self.name(),
                        format!(
                            "epoch regression on page {}: {} < {}",
                            page.page_id.0, page.epoch, epoch.0
                        ),
                    );
                }
                _ => {
                    guard.insert(page.page_id, Epoch(page.epoch));
                }
            }
        }
        InvariantResult::ok(self.name())
    }
}

#[derive(Default)]
pub struct PageConsistency;

impl Invariant for PageConsistency {
    fn name(&self) -> &'static str {
        "PageConsistency"
    }

    fn check(&self, ctx: &InvariantContext) -> InvariantResult {
        let allocator = match ctx.allocator {
            Some(alloc) => alloc,
            None => return InvariantResult::fail(self.name(), "allocator unavailable"),
        };
        for snapshot in allocator.snapshot_pages() {
            if snapshot.data.len() != snapshot.size {
                return InvariantResult::fail(
                    self.name(),
                    format!(
                        "page {} payload mismatch: {} != {}",
                        snapshot.page_id.0,
                        snapshot.data.len(),
                        snapshot.size
                    ),
                );
            }
            if let Err(err) = validate_metadata_blob(&snapshot.metadata_blob) {
                return InvariantResult::fail(
                    self.name(),
                    format!("page {} metadata invalid: {err}", snapshot.page_id.0),
                );
            }
        }
        InvariantResult::ok(self.name())
    }
}

pub struct GraphAcyclicity;

impl GraphAcyclicity {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GraphAcyclicity {
    fn default() -> Self {
        Self::new()
    }
}

impl Invariant for GraphAcyclicity {
    fn name(&self) -> &'static str {
        "GraphAcyclicity"
    }

    fn check(&self, ctx: &InvariantContext) -> InvariantResult {
        let dag = match ctx.graph {
            Some(dag) => dag,
            None => return InvariantResult::fail(self.name(), "graph unavailable"),
        };
        let validator = DagValidator::new(dag);
        let report = validator.detect_cycles();
        if report.has_cycle {
            let ids: Vec<String> = report.cycle.iter().map(|id| id.0.to_string()).collect();
            InvariantResult::fail(
                self.name(),
                format!("cycle detected: {}", ids.join(" → ")),
            )
        } else {
            InvariantResult::ok(self.name())
        }
    }
}

fn validate_metadata_blob(blob: &[u8]) -> Result<(), PageError> {
    if blob.is_empty() {
        return Ok(());
    }
    let mut cursor = 0usize;
    let entry_count = read_u32(blob, &mut cursor)?;
    for _ in 0..entry_count {
        let key_len = read_u32(blob, &mut cursor)? as usize;
        read_bytes(blob, &mut cursor, key_len)?;
        let value_len = read_u32(blob, &mut cursor)? as usize;
        read_bytes(blob, &mut cursor, value_len)?;
    }
    Ok(())
}

fn read_u32(blob: &[u8], cursor: &mut usize) -> Result<u32, PageError> {
    if *cursor + 4 > blob.len() {
        return Err(PageError::MetadataDecode("unexpected end of blob"));
    }
    let val = u32::from_le_bytes(blob[*cursor..*cursor + 4].try_into().unwrap());
    *cursor += 4;
    Ok(val)
}

fn read_bytes(blob: &[u8], cursor: &mut usize, len: usize) -> Result<(), PageError> {
    if *cursor + len > blob.len() {
        return Err(PageError::MetadataDecode("blob truncated"));
    }
    *cursor += len;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{EpochMonotonicity, GraphAcyclicity, InvariantChecker, InvariantContext};
    use crate::dag::{build_dependency_graph, EdgeType, StructuralOp};
    use crate::page::{PageAllocator, PageAllocatorConfig, PageID, PageLocation};
    use crate::page::Epoch;

    #[test]
    fn epoch_invariant_detects_regressions() {
        let allocator = PageAllocator::new(PageAllocatorConfig::default());
        allocator
            .allocate_raw(PageID(1), 8, Some(PageLocation::Cpu))
            .unwrap();
        {
            let page = allocator.acquire_page(PageID(1)).unwrap();
            unsafe { (*page).set_epoch(Epoch(5)); }
        }
        let ctx = InvariantContext {
            allocator: Some(&allocator),
            graph: None,
            registry: None,
        };
        let mut checker = InvariantChecker::new();
        checker.register(EpochMonotonicity::default());
        assert!(checker.run(&ctx)[0].passed);
        {
            let page = allocator.acquire_page(PageID(1)).unwrap();
            unsafe { (*page).set_epoch(Epoch(4)); }
        }
        assert!(!checker.run(&ctx)[0].passed);
    }

    #[test]
    fn graph_acyclicity_detects_cycles() {
        let ops = vec![
            StructuralOp::AddEdge {
                from: PageID(1),
                to: PageID(2),
                edge_type: EdgeType::Data,
            },
            StructuralOp::AddEdge {
                from: PageID(2),
                to: PageID(1),
                edge_type: EdgeType::Data,
            },
        ];
        let dag = build_dependency_graph(&ops);
        let ctx = InvariantContext {
            allocator: None,
            graph: Some(&dag),
            registry: None,
        };
        let mut checker = InvariantChecker::new();
        checker.register(GraphAcyclicity::new());
        assert!(!checker.run(&ctx)[0].passed);
    }
}
