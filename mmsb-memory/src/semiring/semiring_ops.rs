use super::semiring_types::Semiring;

/// Apply semiring addition across an iterator of elements.
pub fn fold_add<S: Semiring>(semiring: &S, values: impl IntoIterator<Item = S::Element>) -> S::Element {
    values
        .into_iter()
        .fold(semiring.zero(), |acc, value| semiring.add(&acc, &value))
}

/// Apply semiring multiplication across an iterator of elements.
pub fn fold_mul<S: Semiring>(semiring: &S, values: impl IntoIterator<Item = S::Element>) -> S::Element {
    values
        .into_iter()
        .fold(semiring.one(), |acc, value| semiring.mul(&acc, &value))
}

/// Convenience helper for combining two elements with both operations.
pub fn accumulate<S: Semiring>(
    semiring: &S,
    left: &S::Element,
    right: &S::Element,
) -> (S::Element, S::Element) {
    (semiring.add(left, right), semiring.mul(left, right))
}
