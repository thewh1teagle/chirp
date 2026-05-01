#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    text: String,
}

impl Prompt {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}
