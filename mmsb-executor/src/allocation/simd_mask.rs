/// Generate a write mask for two equally sized byte slices.
pub fn generate_mask(old: &[u8], new: &[u8]) -> Vec<bool> {
    assert_eq!(old.len(), new.len());
    old.iter().zip(new.iter()).map(|(a, b)| a != b).collect()
}
