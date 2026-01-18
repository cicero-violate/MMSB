#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum EdgeType {
  Data = 0,
  Control = 1,
  Gpu = 2,
  Compiler = 3,
}

impl TryFrom<u8> for EdgeType {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(EdgeType::Data),
            1 => Ok(EdgeType::Control),
            2 => Ok(EdgeType::Gpu),
            3 => Ok(EdgeType::Compiler),
            _ => Err(format!("invalid EdgeType value: {}", value)),
       }
   }
}
