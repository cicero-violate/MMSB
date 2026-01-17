use crate::semiring::{fold_add, fold_mul, Semiring};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct PurityFailure<I, O> {
    pub input: I,
    pub expected: O,
    pub observed: O,
    pub run: usize,
}

#[derive(Debug, Clone)]
pub struct PurityReport<I, O> {
    pub runs: usize,
    pub samples: usize,
    pub failures: Vec<PurityFailure<I, O>>,
}

impl<I, O> PurityReport<I, O> {
    pub fn is_pure(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PurityValidator {
    runs: usize,
}

impl PurityValidator {
    pub fn new(runs: usize) -> Self {
        Self { runs: runs.max(2) }
    }

    pub fn validate_fn<F, I, O>(&self, inputs: &[I], func: F) -> PurityReport<I, O>
    where
        F: Fn(&I) -> O,
        I: Clone + Debug,
        O: Clone + PartialEq + Debug,
    {
        let baseline: Vec<O> = inputs.iter().map(|input| func(input)).collect();
        let mut failures = Vec::new();
        for run in 1..self.runs {
            for (idx, input) in inputs.iter().enumerate() {
                let observed = func(input);
                if observed != baseline[idx] {
                    failures.push(PurityFailure {
                        input: input.clone(),
                        expected: baseline[idx].clone(),
                        observed,
                        run,
                    });
                }
            }
        }
        PurityReport {
            runs: self.runs,
            samples: inputs.len(),
            failures,
        }
    }

    pub fn validate_semiring<S>(&self, semiring: &S, samples: &[Vec<S::Element>]) -> bool
    where
        S: Semiring,
        S::Element: Clone + Debug + PartialEq,
    {
        for values in samples {
            let baseline_add = fold_add(semiring, values.clone());
            let baseline_mul = fold_mul(semiring, values.clone());
            for _ in 1..self.runs {
                if fold_add(semiring, values.clone()) != baseline_add {
                    return false;
                }
                if fold_mul(semiring, values.clone()) != baseline_mul {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for PurityValidator {
    fn default() -> Self {
        PurityValidator::new(3)
    }
}
