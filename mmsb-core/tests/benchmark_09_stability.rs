// Use the public prelude API
struct NoiseRng(u64);

impl NoiseRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((self.0 >> 33) as f64) / (u32::MAX as f64)
    }

    fn gaussian(&mut self, std_dev: f64) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let mag = (-2.0 * u1.ln()).sqrt() * std_dev;
        mag * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn simulate(state: &[f64], drift: f64) -> Vec<f64> {
    state
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let bias = (idx as f64).sin() * 0.001;
            (value + drift + bias).tanh()
        })
        .collect()
}

fn divergence(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn stability_resists_small_noise() {
    let mut rng = NoiseRng::new(42);
    let mut baseline = vec![0.0; 64];
    let mut perturbed = baseline.clone();
    let mut max_divergence: f64 = 0.0;
    for _ in 0..128 {
        baseline = simulate(&baseline, 0.0);
        let noise = rng.gaussian(0.005);
        perturbed = simulate(&perturbed, noise);
        let div = divergence(&baseline, &perturbed);
        assert!(div.is_finite());
        max_divergence = max_divergence.max(div);
    }
    assert!(max_divergence < 1.0);
}
