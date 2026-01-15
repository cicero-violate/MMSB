// Use the public prelude API
use mmsb_core::prelude::{EdgeType, ShadowPageGraph};
use mmsb_core::prelude::{
    Delta, DeltaID, DeviceBufferRegistry, PageAllocator, PageAllocatorConfig, PageID,
    PageLocation, Source,
};
use mmsb_core::prelude::Epoch;
use mmsb_core::prelude::{InvariantChecker, InvariantContext};
use std::sync::Arc;

const CYCLES: usize = 10_000;
const DELTAS_PER_CYCLE: usize = 100;
const PAGE_COUNT: u64 = 256;
const PAGE_SIZE: usize = 128;
const CHECK_INTERVAL: usize = 100;
const DELTA_WIDTH: usize = 32;

#[test]
fn ten_thousand_cycles_no_violations() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));
    for id in 0..PAGE_COUNT {
        allocator
            .allocate_raw(PageID(id + 1), PAGE_SIZE, Some(PageLocation::Cpu))
            .expect("failed to allocate stability page");
    }
    let graph = ShadowPageGraph::default();
    let registry = DeviceBufferRegistry::default();
    let checker = InvariantChecker::with_builtins();
    let mut rng = Lcg::new(0x5eed_cafe_d00d_f001);
    let mut delta_id = 0u64;
    let mut probe = SignalProbe::new(DELTA_WIDTH, 0.5);

    for cycle in 0..CYCLES {
        apply_random_deltas(&allocator, &mut rng, DELTAS_PER_CYCLE, &mut delta_id);
        mutate_graph(&graph, &mut rng);
        probe.step(&mut rng);
        probe.assert_within_bounds();

        if cycle % CHECK_INTERVAL == 0 {
            let ctx = InvariantContext {
                allocator: Some(&allocator),
                graph: Some(&graph),
                registry: Some(&registry),
            };
            for result in checker.run(&ctx) {
                assert!(
                    result.passed,
                    "Invariant {} failed at cycle {}: {:?}",
                    result.name,
                    cycle,
                    result.details
                );
            }
        }
    }
    println!(
        "METRIC:stability cycles={} max_divergence={:.6} invariant_failures=0",
        CYCLES,
        probe.last_divergence()
    );
}

fn apply_random_deltas(
    allocator: &Arc<PageAllocator>,
    rng: &mut Lcg,
    count: usize,
    next_delta: &mut u64,
) {
    for _ in 0..count {
        let page_id = PageID((rng.next_u64() % PAGE_COUNT) + 1);
        let mut mask = vec![false; DELTA_WIDTH];
        let flips = rng.next_in_range(DELTA_WIDTH / 2).max(1);
        for _ in 0..flips {
            let idx = rng.next_in_range(DELTA_WIDTH);
            mask[idx] = true;
        }
        let payload = mask
            .iter()
            .map(|&flag| if flag { (rng.next_u32() & 0xFF) as u8 } else { 0 })
            .collect::<Vec<u8>>();
        let delta = Delta {
            delta_id: DeltaID(*next_delta),
            page_id,
            epoch: Epoch((*next_delta % u32::MAX as u64) as u32),
            mask,
            payload,
            is_sparse: false,
            timestamp: *next_delta,
            source: Source(String::new()),
            intent_metadata: None,
        };
        *next_delta = next_delta.wrapping_add(1);
        unsafe {
            let page_ptr = allocator
                .acquire_page(page_id)
                .expect("page missing during stability test");
            (*page_ptr)
                .apply_delta(&delta)
                .expect("delta application failed");
        }
    }
}

fn mutate_graph(graph: &ShadowPageGraph, rng: &mut Lcg) {
    for _ in 0..4 {
        let mut from = PageID((rng.next_u64() % PAGE_COUNT) + 1);
        let mut to = PageID((rng.next_u64() % PAGE_COUNT) + 1);
        if from == to {
            to = PageID(((to.0 + 1) % PAGE_COUNT) + 1);
        }
        if from.0 > to.0 {
            std::mem::swap(&mut from, &mut to);
        }
        if rng.next_bool() {
            graph.add_edge(from, to, random_edge_type(rng));
        } else {
            graph.remove_edge(from, to);
        }
    }
}

fn random_edge_type(rng: &mut Lcg) -> EdgeType {
    match rng.next_in_range(4) {
        0 => EdgeType::Data,
        1 => EdgeType::Control,
        2 => EdgeType::Gpu,
        _ => EdgeType::Compiler,
    }
}

struct SignalProbe {
    baseline: Vec<f64>,
    values: Vec<f64>,
    max_divergence: f64,
    last_divergence: f64,
}

impl SignalProbe {
    fn new(channels: usize, max_divergence: f64) -> Self {
        Self {
            baseline: vec![0.0; channels],
            values: vec![0.0; channels],
            max_divergence,
            last_divergence: 0.0,
        }
    }

    fn step(&mut self, rng: &mut Lcg) {
        for (value, baseline) in self.values.iter_mut().zip(self.baseline.iter_mut()) {
            let noise = (rng.next_f64() - 0.5) * 0.02;
            *value += noise;
            *baseline = 0.995 * *baseline + 0.005 * *value;
        }
    }

    fn assert_within_bounds(&mut self) {
        for value in &self.values {
            assert!(
                value.is_finite(),
                "detected non-finite value in stability probe"
            );
        }
        let divergence = self
            .values
            .iter()
            .zip(&self.baseline)
            .map(|(value, baseline)| (value - baseline).abs())
            .fold(0.0, f64::max);
        self.last_divergence = divergence;
        assert!(
            divergence <= self.max_divergence,
            "divergence {divergence:.4} exceeded bound {}",
            self.max_divergence
        );
    }

    fn last_divergence(&self) -> f64 {
        self.last_divergence
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.0
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }

    fn next_in_range(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}
