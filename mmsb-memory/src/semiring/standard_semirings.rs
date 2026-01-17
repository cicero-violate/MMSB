use super::semiring_types::Semiring;

#[derive(Clone, Copy, Debug, Default)]
pub struct TropicalSemiring;
impl Semiring for TropicalSemiring {
    type Element = f64;
    fn zero(&self) -> Self::Element {
        f64::INFINITY
    }
    fn one(&self) -> Self::Element {
        0.0
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.min(*b)
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
}
pub struct BooleanSemiring;
impl Semiring for BooleanSemiring {
    type Element = bool;
        false
        true
        *a || *b
        *a && *b
