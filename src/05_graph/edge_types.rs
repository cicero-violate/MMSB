#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    Data = 0,
    Control = 1,
    Gpu = 2,
    Compiler = 3,
}
