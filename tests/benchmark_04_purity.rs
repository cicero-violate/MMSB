use mmsb_core::semiring::{BooleanSemiring, PurityValidator, TropicalSemiring};

#[test]
fn purity_validator_covers_semiring_operations() {
    let validator = PurityValidator::default();
    let boolean_samples = vec![vec![true, false], vec![true, true]];
    assert!(validator.validate_semiring(&BooleanSemiring, &boolean_samples));

    let tropical_samples = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0]];
    assert!(validator.validate_semiring(&TropicalSemiring, &tropical_samples));
}
