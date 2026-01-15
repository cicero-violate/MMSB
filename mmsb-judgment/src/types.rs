#[derive(Debug)]
pub struct JudgmentToken {
    token: String,
    _private: (),
}

impl JudgmentToken {
    pub(crate) fn new(token: String) -> Self {
        Self {
            token,
            _private: (),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    #[cfg(feature = "test-helpers")]
    pub fn test_only() -> Self {
        Self {
            token: "test-only".to_string(),
            _private: (),
        }
    }
}
