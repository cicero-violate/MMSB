/// Core semiring trait used by the propagation engine.
pub trait Semiring: Send + Sync {
    type Element: Clone + PartialEq + Send + Sync;

    /// Additive identity.
    fn zero(&self) -> Self::Element;
    /// Multiplicative identity.
    fn one(&self) -> Self::Element;
    /// Semiring addition (\oplus).
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    /// Semiring multiplication (\otimes).
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}
