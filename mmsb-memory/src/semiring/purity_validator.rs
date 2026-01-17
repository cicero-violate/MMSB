use crate::semiring::semiring_types::Semiring;

pub struct PurityValidator;
impl PurityValidator {
    pub fn validate_semiring<S>(&self, _semiring: &S, _samples: &[Vec<<S as Semiring>::Element>]) -> bool
    where
        S: Semiring,
    {
        true
    }
}
